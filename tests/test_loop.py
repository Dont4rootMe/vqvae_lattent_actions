import json

import torch

from vqvae_latent_actions.training.loop import TrainConfig, lr_lambda, train


def _config(tmp_path, tiny_manifest, eval_path, steps: int) -> TrainConfig:
    return TrainConfig(
        out_dir=str(tmp_path / "run"), manifest=str(tiny_manifest), eval_set=str(eval_path), cache_dir=str(tmp_path / "cache"),
        model={"horizon": 10, "max_horizon": 16, "num_tokens": 4, "dim": 32, "heads": 4, "free_queries": 2,
               "enc_step_layers": 1, "enc_time_layers": 1, "enc_latent_layers": 1, "dec_latent_layers": 1, "dec_layers": 1,
               "quantizer": {"type": "fsq", "levels": [4, 4]}},
        steps=steps, batch_size=4, lr=1e-3, warmup_steps=3, num_workers=0, log_every=1, eval_every=2, ckpt_every=2,
        eval_batch_size=8, mixed_precision="no", comet={"mode": "disabled"}, run_name="unit")


def test_train_resume_and_export(tmp_path, tiny_manifest, tiny_eval_set):
    from vqvae_latent_actions.data.chunks import save_eval_set
    eval_path = tmp_path / "eval.npz"
    save_eval_set(tiny_eval_set, eval_path)

    summary = train(_config(tmp_path, tiny_manifest, eval_path, steps=3))
    run = tmp_path / "run"
    assert summary["total"]["n"] == len(tiny_eval_set)
    assert (run / "checkpoints" / "latest.pt").exists() and (run / "final" / "config.json").exists()
    assert (run / "final_eval.json").exists() and (run / "final_eval.md").exists()
    steps = [json.loads(l)["step"] for l in (run / "train_log.jsonl").read_text().splitlines()]
    assert steps == [1, 2, 3]

    train(_config(tmp_path, tiny_manifest, eval_path, steps=5))          # resumes from the step-3 checkpoint
    steps = [json.loads(l)["step"] for l in (run / "train_log.jsonl").read_text().splitlines()]
    assert steps == [1, 2, 3, 4, 5]

    from vqvae_latent_actions.models.hier_tokenizer import HierActionTokenizer
    model = HierActionTokenizer.from_pretrained(run / "final")
    assert model.num_tokens == 4 and model.vocab_size == 16


def test_lr_follows_the_schedule_every_step(tmp_path, tiny_manifest, tiny_eval_set):
    from vqvae_latent_actions.data.chunks import save_eval_set
    eval_path = tmp_path / "eval.npz"
    save_eval_set(tiny_eval_set, eval_path)
    cfg = _config(tmp_path, tiny_manifest, eval_path, steps=6)   # warmup 3 then cosine decay, so lr rises and falls
    train(cfg)
    rows = [json.loads(l) for l in (tmp_path / "run" / "train_log.jsonl").read_text().splitlines()]
    schedule = lr_lambda(cfg)
    for row in rows:
        assert abs(row["lr"] - cfg.lr * schedule(row["step"])) < 1e-9
    assert rows[0]["lr"] < rows[1]["lr"] and rows[-1]["lr"] < rows[1]["lr"]


def test_param_groups_can_exempt_named_parameters_from_weight_decay(tiny_model_config):
    """Weight decay on the projection that reads the code pushes the decoder to want larger codes; exempting the
    code projections tests whether that is what drives the latent scale up."""
    from vqvae_latent_actions.models.hier_tokenizer import HierActionTokenizer
    from vqvae_latent_actions.training.loop import param_groups
    model = HierActionTokenizer(tiny_model_config)
    targets = {id(model.from_code.weight), id(model.to_code.weight)}

    default = param_groups(model, 0.01)
    assert targets <= {id(p) for p in default[0]["params"]} and default[0]["weight_decay"] == 0.01

    exempt = param_groups(model, 0.01, ["from_code", "to_code"])
    assert targets <= {id(p) for p in exempt[1]["params"]} and exempt[1]["weight_decay"] == 0.0
    trainable = [id(p) for p in model.parameters() if p.requires_grad]
    grouped = [id(p) for group in exempt for p in group["params"]]
    assert sorted(grouped) == sorted(trainable)                      # every parameter exactly once


def test_resume_drops_log_rows_from_an_attempt_that_never_checkpointed(tmp_path, tiny_manifest, tiny_eval_set):
    """Low-priority jobs are stopped and rerun by the queue. Steps logged after the last checkpoint are trained
    again on the rerun, so their old rows must go, or the log shows every such step twice."""
    from vqvae_latent_actions.data.chunks import save_eval_set
    eval_path = tmp_path / "eval.npz"
    save_eval_set(tiny_eval_set, eval_path)
    train(_config(tmp_path, tiny_manifest, eval_path, steps=3))          # checkpoint at step 3
    log = tmp_path / "run" / "train_log.jsonl"
    with log.open("a") as handle:                                       # the preempted attempt got further
        for step in (4, 5):
            handle.write(json.dumps({"step": step, "loss": 9.9}) + "\n")
    train(_config(tmp_path, tiny_manifest, eval_path, steps=5))
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["step"] for r in rows] == [1, 2, 3, 4, 5]
    assert all(r["loss"] != 9.9 for r in rows)


def test_scale_diagnostics_report_what_drifts(tiny_model_config):
    from vqvae_latent_actions.models.hier_tokenizer import HierActionTokenizer, HierTokenizerConfig
    from vqvae_latent_actions.training.loop import scale_diagnostics
    latents = torch.randn(2, 4, 2) * 3
    fsq = scale_diagnostics(HierActionTokenizer(tiny_model_config), latents)
    assert set(fsq) == {"code_norm", "from_code_norm"} and all(v > 0 for v in fsq.values())
    vq_cfg = HierTokenizerConfig.from_dict({**tiny_model_config.to_dict(),
                                            "quantizer": {"type": "vq", "vocab_size": 16, "code_dim": 2}})
    vq = scale_diagnostics(HierActionTokenizer(vq_cfg), latents)
    assert "codebook_norm" in vq and vq["code_norm"] > 0
