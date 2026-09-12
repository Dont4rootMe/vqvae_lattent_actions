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
