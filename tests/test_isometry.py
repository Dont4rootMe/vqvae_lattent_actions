from __future__ import annotations

import math

import pytest
import torch


def _linear_decoder(matrix):
    """decode(z) -> [B, T, D] for a fixed linear map, so the penalty has a closed form."""
    t, d = 2, matrix.shape[0] // 2

    def decode(z):
        flat = z.reshape(z.shape[0], -1) @ matrix.T
        return flat.reshape(z.shape[0], t, d)
    return decode


def _traces(matrix, weights=None):
    """Analytic Tr(G) and Tr(G^2) for G = A^T H A."""
    h = torch.ones(matrix.shape[0]) if weights is None else weights
    g = matrix.T @ torch.diag(h) @ matrix
    return float(torch.diagonal(g).sum()), float((g @ g).diagonal().sum())


def test_estimate_matches_the_analytic_traces_for_a_linear_map():
    from vqvae_latent_actions.training.isometry import IsometryConfig, isometry_penalty
    torch.manual_seed(0)
    m, out = 8, 12
    a = torch.randn(out, m)
    z = torch.randn(4096, m // 4, 4)
    cfg = IsometryConfig(weight=1.0, subbatch=4096, alpha_jitter=0.0)
    got = isometry_penalty(_linear_decoder(a), z, None, cfg, generator=torch.Generator().manual_seed(1))
    tr_g, tr_g2 = _traces(a)
    assert got["trace"] == pytest.approx(tr_g, rel=0.05)
    assert got["trace_squared"] == pytest.approx(tr_g2, rel=0.15)


def test_a_conformal_map_sits_at_the_minimum_and_uses_every_direction():
    from vqvae_latent_actions.training.isometry import IsometryConfig, isometry_penalty
    torch.manual_seed(0)
    m = 8
    q, _ = torch.linalg.qr(torch.randn(m, m))
    z = torch.randn(4096, m // 4, 4)
    cfg = IsometryConfig(weight=1.0, subbatch=4096, alpha_jitter=0.0)
    out = isometry_penalty(_linear_decoder(3.0 * q), z, None, cfg, generator=torch.Generator().manual_seed(1))
    assert out["rdm"] == pytest.approx(1.0 / m, rel=0.1)
    assert out["participation_ratio"] == pytest.approx(m, rel=0.1)


def test_an_anisotropic_map_is_penalized_more():
    from vqvae_latent_actions.training.isometry import IsometryConfig, isometry_penalty
    torch.manual_seed(0)
    m = 8
    q, _ = torch.linalg.qr(torch.randn(m, m))
    stretched = q @ torch.diag(torch.tensor([8.0, 4.0] + [0.2] * (m - 2)))
    z = torch.randn(4096, m // 4, 4)
    cfg = IsometryConfig(weight=1.0, subbatch=4096, alpha_jitter=0.0)
    gen = torch.Generator().manual_seed(1)
    flat = isometry_penalty(_linear_decoder(q), z, None, cfg, generator=gen)
    sharp = isometry_penalty(_linear_decoder(stretched), z, None, cfg, generator=gen)
    assert sharp["rdm"] > 3 * flat["rdm"]
    assert sharp["participation_ratio"] < flat["participation_ratio"] / 3


def test_the_measure_ignores_the_overall_scale_of_the_decoder():
    from vqvae_latent_actions.training.isometry import IsometryConfig, isometry_penalty
    torch.manual_seed(0)
    a = torch.randn(12, 8)
    z = torch.randn(2048, 2, 4)
    cfg = IsometryConfig(weight=1.0, subbatch=2048, alpha_jitter=0.0)
    small = isometry_penalty(_linear_decoder(a), z, None, cfg, generator=torch.Generator().manual_seed(2))
    large = isometry_penalty(_linear_decoder(100.0 * a), z, None, cfg, generator=torch.Generator().manual_seed(2))
    assert large["rdm"] == pytest.approx(small["rdm"], rel=1e-3)


def test_weights_drop_the_padded_outputs(monkeypatch):
    from vqvae_latent_actions.training.isometry import IsometryConfig, isometry_penalty
    torch.manual_seed(0)
    a = torch.randn(12, 8)
    z = torch.randn(2048, 2, 4)
    cfg = IsometryConfig(weight=1.0, subbatch=2048, alpha_jitter=0.0)
    weights = torch.ones(2048, 2, 6)
    weights[:, :, 3:] = 0.0                       # the second half of every step is padding
    got = isometry_penalty(_linear_decoder(a), z, weights, cfg, generator=torch.Generator().manual_seed(3))
    kept = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0] * 2)
    tr_g, tr_g2 = _traces(a, kept)
    assert got["trace"] == pytest.approx(tr_g, rel=0.05)
    assert got["trace_squared"] == pytest.approx(tr_g2, rel=0.15)


def test_the_penalty_trains_the_decoder_with_a_single_backward():
    from vqvae_latent_actions.training.isometry import IsometryConfig, isometry_penalty
    torch.manual_seed(0)
    linear = torch.nn.Linear(8, 12, bias=False)

    def decode(z):
        return linear(z.reshape(z.shape[0], -1)).reshape(z.shape[0], 2, 6)

    z = torch.randn(256, 2, 4)
    out = isometry_penalty(decode, z, None, IsometryConfig(weight=0.5, subbatch=256), generator=None)
    out["loss"].backward()
    assert linear.weight.grad is not None and torch.isfinite(linear.weight.grad).all()
    assert float(linear.weight.grad.abs().sum()) > 0.0


def test_subbatch_limits_how_many_chunks_are_probed():
    from vqvae_latent_actions.training.isometry import IsometryConfig, isometry_penalty
    seen = []

    def decode(z):
        seen.append(z.shape[0])
        return z.reshape(z.shape[0], -1).unsqueeze(1).expand(-1, 2, -1).clone()

    z = torch.randn(512, 2, 4)
    isometry_penalty(decode, z, None, IsometryConfig(weight=1.0, subbatch=32), generator=None)
    assert set(seen) == {32}


def test_disabled_config_is_free_and_returns_nothing():
    from vqvae_latent_actions.training.isometry import IsometryConfig, isometry_penalty
    calls = []

    def decode(z):
        calls.append(1)
        return z
    assert not IsometryConfig().enabled
    assert isometry_penalty(decode, torch.randn(4, 2, 4), None, IsometryConfig(), generator=None) is None
    assert not calls


def test_model_forward_reports_the_penalty_and_leaves_the_plain_loss_alone(tiny_model_config):
    from vqvae_latent_actions.models.hier_tokenizer import HierActionTokenizer
    from vqvae_latent_actions.training.isometry import IsometryConfig
    torch.manual_seed(0)
    model = HierActionTokenizer(tiny_model_config)
    actions = torch.randn(8, 10, 119)
    mask = torch.ones_like(actions, dtype=torch.bool)
    mask[:, :, 50:] = False
    plain = model(actions, mask)
    with_iso = model(actions, mask, isometry=IsometryConfig(weight=1.0, subbatch=4))
    assert "iso_loss" not in plain
    assert float(with_iso["recon_mse"]) == pytest.approx(float(plain["recon_mse"]), rel=1e-5)
    assert float(with_iso["loss"]) > float(with_iso["recon_mse"])
    assert with_iso["iso_participation_ratio"] > 0


def test_training_with_the_penalty_runs_and_logs_it(tmp_path, tiny_manifest, tiny_eval_set):
    import json
    from vqvae_latent_actions.data.chunks import save_eval_set
    from vqvae_latent_actions.training.loop import TrainConfig, train
    eval_path = tmp_path / "eval.npz"
    save_eval_set(tiny_eval_set, eval_path)
    cfg = TrainConfig(
        out_dir=str(tmp_path / "run"), manifest=str(tiny_manifest), eval_set=str(eval_path),
        cache_dir=str(tmp_path / "cache"),
        model={"horizon": 10, "max_horizon": 16, "num_tokens": 4, "dim": 32, "heads": 4, "free_queries": 2,
               "enc_step_layers": 1, "enc_time_layers": 1, "enc_latent_layers": 1, "dec_latent_layers": 1,
               "dec_layers": 1, "quantizer": {"type": "vq", "vocab_size": 16, "code_dim": 4, "cosine": True}},
        steps=3, batch_size=4, lr=1e-3, warmup_steps=3, num_workers=0, log_every=1, eval_every=3, ckpt_every=3,
        eval_batch_size=8, mixed_precision="no", comet={"mode": "disabled"}, run_name="iso",
        isometry={"weight": 1.0, "subbatch": 2})
    train(cfg)
    rows = [json.loads(l) for l in (tmp_path / "run" / "train_log.jsonl").read_text().splitlines()]
    assert all(row["iso_participation_ratio"] > 0 for row in rows)
    assert all(row["iso_loss"] > 0 for row in rows)


def test_the_penalty_is_skipped_during_its_warmup():
    from vqvae_latent_actions.training.isometry import IsometryConfig
    cfg = IsometryConfig(weight=1.0, warmup_steps=10, every=2)
    assert not cfg.active(4) and not cfg.active(11)
    assert cfg.active(10) and cfg.active(12)


def test_config_reaches_the_train_config():
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    from train import build_train_config
    from vqvae_latent_actions.training.isometry import IsometryConfig
    root = Path(__file__).resolve().parent.parent
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        default = compose(config_name="config", overrides=["run_name=unit"])
        on = compose(config_name="config", overrides=["run_name=unit", "train.isometry.weight=3.0",
                                                      "train.isometry.every=4"])
    assert not IsometryConfig.from_dict(build_train_config(default).isometry).enabled
    cfg = IsometryConfig.from_dict(build_train_config(on).isometry)
    assert cfg.weight == 3.0 and cfg.every == 4
