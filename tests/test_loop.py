import json

import pytest
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

    with pytest.raises(TypeError, match="list"):
        param_groups(model, 0.01, "from_code")                          # a bare string would match every name

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
        handle.write('{"step": 6, "lo')                               # killed in the middle of a write
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



def test_rerun_without_a_checkpoint_starts_the_logs_over(tmp_path, tiny_manifest, tiny_eval_set):
    """Stopped before its first checkpoint, the rerun trains from step 0 again: every old row is stale, and the first
    new row must not be glued onto a line the kill cut short."""
    from vqvae_latent_actions.data.chunks import save_eval_set
    eval_path = tmp_path / "eval.npz"
    save_eval_set(tiny_eval_set, eval_path)
    run = tmp_path / "run"
    run.mkdir(parents=True)
    log = run / "train_log.jsonl"
    log.write_text('{"step": 1, "loss": 9.9}\n{"step": 2, "loss": 9.9}\n{"step": 3, "lo')
    train(_config(tmp_path, tiny_manifest, eval_path, steps=2))
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["step"] for r in rows] == [1, 2] and all(r["loss"] != 9.9 for r in rows)


def test_rerun_of_a_finished_run_evaluates_the_last_step_once_and_keeps_counting_data(tmp_path, tiny_manifest,
                                                                                      tiny_eval_set):
    from vqvae_latent_actions.data.chunks import save_eval_set
    eval_path = tmp_path / "eval.npz"
    save_eval_set(tiny_eval_set, eval_path)
    cfg = _config(tmp_path, tiny_manifest, eval_path, steps=3)
    train(cfg)
    train(cfg)                                                        # killed during export, rerun as is
    run = tmp_path / "run"
    evals = [json.loads(line)["step"] for line in (run / "eval_log.jsonl").read_text().splitlines()]
    assert evals.count(3) == 1
    train(_config(tmp_path, tiny_manifest, eval_path, steps=5))       # resumed at 3: data seen keeps growing
    rows = {r["step"]: r for r in map(json.loads, (run / "train_log.jsonl").read_text().splitlines())}
    assert rows[4]["chunks_seen"] == 4 * cfg.batch_size and rows[5]["chunks_seen"] == 5 * cfg.batch_size



def test_snapshots_best_checkpoint_and_stability_signals(tmp_path, tiny_manifest, tiny_eval_set):
    """A 10-token production run reached its best state at step 200k and broke by 220k; with only latest.pt kept the
    good weights were gone. Keep periodic snapshots and the best evaluation, and let a rerun keep the best."""
    from dataclasses import replace
    from vqvae_latent_actions.data.chunks import save_eval_set
    from vqvae_latent_actions.models.hier_tokenizer import HierActionTokenizer
    eval_path = tmp_path / "eval.npz"
    save_eval_set(tiny_eval_set, eval_path)
    cfg = replace(_config(tmp_path, tiny_manifest, eval_path, steps=6), snapshot_every=3)
    train(cfg)
    run = tmp_path / "run"
    assert sorted(p.name for p in (run / "snapshots").iterdir()) == ["step_0000003", "step_0000006"]
    HierActionTokenizer.from_pretrained(run / "snapshots" / "step_0000003")

    evals = [json.loads(line) for line in (run / "eval_log.jsonl").read_text().splitlines()]
    best = json.loads((run / "best" / "best.json").read_text())
    assert best["rmse"] == min(e["eval/rmse"] for e in evals)
    assert best["step"] in {e["step"] for e in evals}
    HierActionTokenizer.from_pretrained(run / "best")
    assert all("eval/attn_max_logit" in e for e in evals)

    rows = [json.loads(line) for line in (run / "train_log.jsonl").read_text().splitlines()]
    assert all("grad_norm" in r and "codebook_restarts" in r for r in rows)

    latest = run / "checkpoints" / "latest.pt"
    payload = torch.load(latest, weights_only=False)
    assert payload["best_rmse"] == best["rmse"]
    payload["best_rmse"] = 0.0                             # a rerun must not replace a better best with a worse one
    torch.save(payload, latest)
    train(replace(cfg, steps=8))
    assert json.loads((run / "best" / "best.json").read_text()) == best



def test_a_kill_during_a_snapshot_export_is_redone_on_the_rerun(tmp_path, tiny_manifest, tiny_eval_set, monkeypatch):
    """The snapshot used to be exported after latest.pt for the same step: a kill during the export resumed past that
    step and the snapshot was never written."""
    from dataclasses import replace
    import pytest
    from vqvae_latent_actions.data.chunks import save_eval_set
    from vqvae_latent_actions.models.hier_tokenizer import HierActionTokenizer
    eval_path = tmp_path / "eval.npz"
    save_eval_set(tiny_eval_set, eval_path)
    cfg = replace(_config(tmp_path, tiny_manifest, eval_path, steps=6), snapshot_every=4, ckpt_every=4)
    original = HierActionTokenizer.save_pretrained

    def killed(self, directory):
        if str(directory).endswith("step_0000004"):
            raise KeyboardInterrupt("killed during the snapshot export")
        return original(self, directory)

    monkeypatch.setattr(HierActionTokenizer, "save_pretrained", killed)
    with pytest.raises(KeyboardInterrupt):
        train(cfg)
    monkeypatch.setattr(HierActionTokenizer, "save_pretrained", original)
    train(cfg)
    HierActionTokenizer.from_pretrained(tmp_path / "run" / "snapshots" / "step_0000004")


def test_rerun_keeps_a_best_export_newer_than_its_checkpoint(tmp_path, tiny_manifest, tiny_eval_set):
    """best/ is exported during eval, before latest.pt for that step. A kill in between resumes from an older
    checkpoint whose best_rmse does not know about that export; best.json must be read too."""
    from dataclasses import replace
    from vqvae_latent_actions.data.chunks import save_eval_set
    eval_path = tmp_path / "eval.npz"
    save_eval_set(tiny_eval_set, eval_path)
    cfg = _config(tmp_path, tiny_manifest, eval_path, steps=4)
    train(cfg)
    run = tmp_path / "run"
    marker = run / "best" / "best.json"
    marker.write_text(json.dumps({"step": 4, "rmse": 0.0}))                 # an export no evaluation can beat
    latest = run / "checkpoints" / "latest.pt"
    payload = torch.load(latest, weights_only=False)
    payload["best_rmse"] = float("inf")                                    # the checkpoint predates that export
    torch.save(payload, latest)
    train(replace(cfg, steps=6))
    assert json.loads(marker.read_text()) == {"step": 4, "rmse": 0.0}
