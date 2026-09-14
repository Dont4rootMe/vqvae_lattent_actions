"""Step-based trainer: accelerate DDP, VLA-weighted sampling, periodic eval on the shared set, resume, export."""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from ..data.chunks import batch_to_inputs, layout_from_manifest, load_eval_set, train_loader
from ..models.hier_tokenizer import HierActionTokenizer, HierTokenizerConfig
from .comet import RunLogger
from .evaluate import evaluate_tokenizer, model_report_extra, padding_invariance_mismatch
from .metrics import write_report


@dataclass
class TrainConfig:
    out_dir: str
    manifest: str
    eval_set: str
    cache_dir: str | None = None
    model: dict[str, Any] = field(default_factory=dict)
    steps: int = 300_000
    batch_size: int = 256
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 2000
    schedule: str = "cosine"
    grad_clip: float = 1.0
    num_workers: int = 10
    samples_per_episode: int = 8
    seed: int = 0
    log_every: int = 100
    eval_every: int = 10_000
    ckpt_every: int = 5_000
    eval_batch_size: int = 1024
    mixed_precision: str = "bf16"
    quantizer_warmup_steps: int = 0   # train the plain autoencoder first, then switch the quantizer on
    # parameters whose name contains any of these get no weight decay (in addition to every 1-d parameter)
    no_decay_keywords: list[str] = field(default_factory=list)
    comet: dict[str, Any] = field(default_factory=dict)
    run_name: str = "hier"


def lr_lambda(cfg: TrainConfig):
    def factor(step: int) -> float:
        if step < cfg.warmup_steps:
            return (step + 1) / max(1, cfg.warmup_steps)
        if cfg.schedule == "constant":
            return 1.0
        progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps))
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return factor


def param_groups(model: torch.nn.Module, weight_decay: float, no_decay_keywords=()) -> list[dict]:
    """AdamW groups: decay for matrices, none for 1-d parameters or names matching a keyword."""
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        exempt = parameter.dim() < 2 or any(keyword in name for keyword in no_decay_keywords)
        (no_decay if exempt else decay).append(parameter)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


@torch.no_grad()
def scale_diagnostics(model: HierActionTokenizer, latents: torch.Tensor) -> dict[str, float]:
    """The quantities that drift when nothing anchors the code scale: the code itself, the projection that reads
    it, and the codebook it is matched against."""
    out = {"code_norm": float(latents.detach().float().norm(dim=-1).mean()),
           "from_code_norm": float(model.from_code.weight.detach().float().norm())}
    codebook = getattr(model.quantizer, "codebook", None)
    if isinstance(codebook, torch.Tensor):
        out["codebook_norm"] = float(codebook.float().norm(dim=-1).mean())
    return out


def truncate_log(path: Path, step: int) -> None:
    """Keep rows up to `step`: a rerun trains every later step again and would log it twice."""
    if not path.exists():
        return
    kept = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:              # the job was killed in the middle of writing this line
            continue
        if int(row.get("step", -1)) <= step:
            kept.append(line)
    tmp = path.with_name(path.name + ".tmp")      # a second kill during the rewrite must not lose the log
    tmp.write_text("".join(line + "\n" for line in kept))
    tmp.replace(path)


def build_model(cfg: TrainConfig, layout) -> HierActionTokenizer:
    return HierActionTokenizer(HierTokenizerConfig.from_dict({**cfg.model, "layout": layout.to_dict()}))


def train(cfg: TrainConfig) -> dict:
    from accelerate import Accelerator, DistributedDataParallelKwargs
    from accelerate.utils import set_seed

    # step_scheduler_with_optimizer=False: accelerate would otherwise advance the schedule once per process.
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision if cfg.mixed_precision != "no" else "no",
                              step_scheduler_with_optimizer=False,
                              kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False,
                                                                            broadcast_buffers=True)])
    rank, world = accelerator.process_index, accelerator.num_processes
    device = accelerator.device
    set_seed(cfg.seed, device_specific=True)
    out = Path(cfg.out_dir)
    if accelerator.is_main_process:
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(asdict(cfg), indent=1, default=str))

    layout = layout_from_manifest(cfg.manifest)
    eval_set = load_eval_set(cfg.eval_set) if accelerator.is_main_process else None
    model = build_model(cfg, layout).to(device)
    optimizer = torch.optim.AdamW(param_groups(model, cfg.weight_decay, cfg.no_decay_keywords),
                                  lr=cfg.lr, betas=tuple(cfg.betas))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(cfg))

    step = 0
    latest = out / "checkpoints" / "latest.pt"
    if latest.exists():
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        step = int(payload["step"])
        accelerator.print(f"resumed from {latest} at step {step}")
        if accelerator.is_main_process:          # a preempted attempt may have logged past its last checkpoint
            for log_path in (out / "train_log.jsonl", out / "eval_log.jsonl"):
                truncate_log(log_path, step)

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    loader = train_loader(cfg.manifest, cfg.cache_dir, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                          seed=cfg.seed * 1000 + rank, samples_per_episode=cfg.samples_per_episode,
                          epoch_length=cfg.batch_size * (cfg.steps + 1), pin_memory=device.type == "cuda")
    loader.dataset.set_epoch(step)

    logger = RunLogger(enabled=accelerator.is_main_process, jsonl_path=out / "train_log.jsonl",
                       eval_jsonl_path=out / "eval_log.jsonl",
                       experiment_name=cfg.run_name, config={"train": asdict(cfg)},
                       resume_key_path=out / "comet_experiment_key",
                       mode=str(cfg.comet.get("mode", "auto")), project=cfg.comet.get("project"),
                       workspace=cfg.comet.get("workspace"), tags=cfg.comet.get("tags"),
                       offline_directory=out / "comet_offline")
    if accelerator.is_main_process:
        extra = model_report_extra(accelerator.unwrap_model(model))
        logger.log_params({"model": extra})
        accelerator.print(f"model: {extra['parameters'] / 1e6:.1f}M params, {extra['num_tokens']} tokens x "
                          f"{extra['vocab_size']} codes = {extra['bits_per_chunk']:.0f} bits/chunk")

    def run_eval(current: int) -> dict:
        target = accelerator.unwrap_model(model)
        # the warmup trains a plain autoencoder, so until it is over the grid is not what the model reconstructs from
        quantized = current > cfg.quantizer_warmup_steps
        summary = evaluate_tokenizer(target, eval_set, batch_size=cfg.eval_batch_size, device=device,
                                     quantize=quantized)
        summary["padding_invariance_mismatch"] = padding_invariance_mismatch(target, eval_set, device=device)
        write_report(out / f"eval_step{current:07d}.json", f"{cfg.run_name}@{current}", summary,
                     {"step": current, **model_report_extra(target)})
        total, usage = summary["total"], summary["usage"]
        logger.log_metrics({"eval/rmse": total["rmse"], "eval/l1": total["l1"], "eval/max_abs": total["max_abs_mean"],
                            "eval/codes_used": usage["codes_used"], "eval/perplexity": usage["perplexity"],
                            "eval/min_position_perplexity": usage["min_position_perplexity"],
                            "eval/effective_bits": usage["effective_bits_per_chunk"],
                            "eval/quantized": int(quantized),
                            "eval/padding_mismatch": summary["padding_invariance_mismatch"]}, step=current, split="eval")
        codes = (f"codes={usage['codes_used']}/{usage['vocab_size']} perplexity={usage['perplexity']:.0f} "
                 f"min_pos_ppl={usage['min_position_perplexity']:.1f} "
                 f"bits={usage['effective_bits_per_chunk']:.0f}/{usage['bits_per_chunk']:.0f}"
                 if quantized else "continuous (quantizer warmup)")
        accelerator.print(f"eval step {current}: rmse={total['rmse']:.5f} l1={total['l1']:.5f} {codes}")
        return summary

    def save_checkpoint() -> None:
        if not accelerator.is_main_process:
            return
        directory = out / "checkpoints"
        directory.mkdir(parents=True, exist_ok=True)
        payload = {"step": step, "model": accelerator.unwrap_model(model).state_dict(),
                   "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "config": asdict(cfg)}
        tmp = directory / "latest.tmp"
        torch.save(payload, tmp)
        tmp.replace(directory / "latest.pt")

    model.train()
    start_time = last_log = time.time()
    seen = 0
    last_summary: dict = {}
    iterator = iter(loader)
    while step < cfg.steps:
        try:
            batch = next(iterator)
        except StopIteration:
            loader.dataset.set_epoch(step + 1)
            iterator = iter(loader)
            batch = next(iterator)
        actions, mask, _ = batch_to_inputs(batch)
        actions, mask = actions.to(device, non_blocking=True), mask.to(device, non_blocking=True)
        quantize = step >= cfg.quantizer_warmup_steps
        with accelerator.autocast():
            output = model(actions, mask, quantize=quantize)
        accelerator.backward(output["loss"])
        if cfg.grad_clip:
            accelerator.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        seen += cfg.batch_size * world

        if step % cfg.log_every == 0:
            loss = accelerator.gather(output["loss"].detach().float().reshape(1)).mean().item()
            latents = output["latents"].detach()
            record = {"step": step, "loss": loss, "recon_mse": float(output["recon_mse"]),
                      "aux_loss": float(output["aux_loss"]), "quantized": int(quantize),
                      # an encoder that drifts out of the quantizer's range silently loses most of the vocabulary
                      "latent_abs_mean": float(latents.abs().mean()),
                      "latent_saturation": accelerator.unwrap_model(model).quantizer.saturation(latents),
                      **scale_diagnostics(accelerator.unwrap_model(model), latents),
                      "lr": scheduler.get_last_lr()[0],
                      "chunks_seen": seen, "steps_per_s": cfg.log_every / max(time.time() - last_log, 1e-9),
                      "elapsed_s": time.time() - start_time}
            last_log = time.time()
            if accelerator.is_main_process:
                logger.log_metrics(record, step=step)
                accelerator.print(f"step {step}/{cfg.steps} loss={loss:.5f} recon={record['recon_mse']:.5f} "
                                  f"lr={record['lr']:.2e} {record['steps_per_s']:.2f} it/s")

        if step % cfg.eval_every == 0 or step == cfg.steps:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                last_summary = run_eval(step)
                model.train()
            accelerator.wait_for_everyone()
        if step % cfg.ckpt_every == 0 or step == cfg.steps:
            save_checkpoint()
            accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        target = accelerator.unwrap_model(model)
        export = target.save_pretrained(out / "final")
        if not last_summary:
            last_summary = run_eval(step)
        write_report(out / "final_eval.json", cfg.run_name, last_summary,
                     {"step": step, "export": str(export), **model_report_extra(target)})
        logger.log_other("export", str(export))
        logger.end()
        accelerator.print(f"final export: {export}")
    accelerator.wait_for_everyone()
    return last_summary


__all__ = ["TrainConfig", "train", "build_model", "lr_lambda"]
