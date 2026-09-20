from __future__ import annotations

import json

import numpy as np
import pytest
import torch


@pytest.fixture
def model(tiny_model_config):
    from vqvae_latent_actions.models.hier_tokenizer import HierActionTokenizer
    torch.manual_seed(0)
    return HierActionTokenizer(tiny_model_config).eval()


def test_gaussian_noise_leaves_padding_untouched(tiny_eval_set):
    from vqvae_latent_actions.ood.perturb import gaussian_noise
    out = gaussian_noise(tiny_eval_set.actions, tiny_eval_set.valid, sigma=0.5, seed=0)
    assert out.valid is tiny_eval_set.valid or np.array_equal(out.valid, tiny_eval_set.valid)
    assert np.all(out.actions[~tiny_eval_set.valid] == 0.0)
    assert np.any(out.actions[tiny_eval_set.valid] != tiny_eval_set.actions[tiny_eval_set.valid])


def test_zero_sigma_is_the_identity(tiny_eval_set):
    from vqvae_latent_actions.ood.perturb import gaussian_noise
    out = gaussian_noise(tiny_eval_set.actions, tiny_eval_set.valid, sigma=0.0, seed=0)
    assert np.array_equal(out.actions, tiny_eval_set.actions)


def test_amplitude_scale_scales_only_real_entries(tiny_eval_set):
    from vqvae_latent_actions.ood.perturb import amplitude_scale
    out = amplitude_scale(tiny_eval_set.actions, tiny_eval_set.valid, factor=2.0)
    v = tiny_eval_set.valid
    assert np.allclose(out.actions[v], 2.0 * tiny_eval_set.actions[v])
    assert np.all(out.actions[~v] == 0.0)


def test_time_stretch_of_one_is_the_identity(tiny_eval_set):
    from vqvae_latent_actions.ood.perturb import time_stretch
    out = time_stretch(tiny_eval_set.actions, tiny_eval_set.valid, factor=1.0)
    assert np.allclose(out.actions, tiny_eval_set.actions)
    assert np.array_equal(out.valid, tiny_eval_set.valid)


def test_time_stretch_keeps_shape_and_padding(tiny_eval_set):
    from vqvae_latent_actions.ood.perturb import time_stretch
    out = time_stretch(tiny_eval_set.actions, tiny_eval_set.valid, factor=2.0)
    assert out.actions.shape == tiny_eval_set.actions.shape
    assert np.all(out.actions[~out.valid] == 0.0)


def test_drop_group_removes_the_group_from_the_mask(tiny_eval_set, layout):
    from vqvae_latent_actions.ood.perturb import drop_group
    name = layout.names[0]
    start, end = layout.intervals()[name]
    out = drop_group(tiny_eval_set.actions, tiny_eval_set.valid, layout, name)
    assert not out.valid[:, :, start:end].any()
    assert np.all(out.actions[:, :, start:end] == 0.0)
    rest = slice(end, None)
    assert np.array_equal(out.valid[:, :, rest], tiny_eval_set.valid[:, :, rest])


def test_dropped_group_decodes_to_exact_zeros(model, tiny_eval_set, layout):
    from vqvae_latent_actions.ood.perturb import drop_group
    name = layout.names[0]
    start, end = layout.intervals()[name]
    out = drop_group(tiny_eval_set.actions, tiny_eval_set.valid, layout, name)
    _, recon = model.encode_decode(torch.as_tensor(out.actions), torch.as_tensor(out.valid))
    assert torch.all(recon[:, :, start:end] == 0.0)


def test_noise_sweep_has_one_row_per_sigma_and_is_exact_at_zero(model, tiny_eval_set):
    from vqvae_latent_actions.ood.robustness import noise_sweep
    rows = noise_sweep(model, tiny_eval_set, sigmas=[0.0, 0.5], batch_size=4)
    assert [r["sigma"] for r in rows] == [0.0, 0.5]
    assert rows[0]["tokens_identical"] == 1.0 and rows[0]["token_agreement"] == 1.0
    assert rows[0]["rmse"] == pytest.approx(rows[0]["clean_rmse"])


def test_token_agreement_counts_positions_and_chunks():
    from vqvae_latent_actions.ood.robustness import token_agreement
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[1, 2, 9], [4, 5, 6]])
    out = token_agreement(a, b)
    assert out["tokens_identical"] == 0.5
    assert out["token_agreement"] == pytest.approx(5 / 6)


def test_latent_interpolation_reports_travel_between_endpoints(model, tiny_eval_set):
    from vqvae_latent_actions.ood.robustness import latent_interpolation
    out = latent_interpolation(model, tiny_eval_set, num_pairs=3, seed=0)
    assert 0.0 <= out["monotone_fraction"] <= 1.0
    assert out["midpoint_ratio"] > 0.0


def test_report_writes_json_and_markdown(model, tiny_eval_set, tmp_path):
    from vqvae_latent_actions.ood.report import run_suite, write_report
    result = run_suite(model, tiny_eval_set, sigmas=[0.0, 0.5], factors=[2.0], time_factors=[2.0],
                       batch_size=4, num_pairs=3)
    paths = write_report(result, tmp_path / "ood", name="tiny")
    payload = json.loads((tmp_path / "ood" / "tiny.json").read_text())
    assert payload["clean"]["rmse"] == pytest.approx(result["clean"]["rmse"])
    assert {"noise", "amplitude", "time", "groups", "interpolation"} <= set(payload)
    assert (tmp_path / "ood" / "tiny.md").read_text().startswith("# tiny")
    assert set(paths) == {"json", "markdown"}


def test_cli_writes_a_report_for_an_exported_model(model, tiny_eval_set, tmp_path):
    from vqvae_latent_actions.data.chunks import save_eval_set
    from vqvae_latent_actions.ood.__main__ import main
    model.save_pretrained(tmp_path / "final")
    save_eval_set(tiny_eval_set, tmp_path / "eval.npz")
    result = main(["--model", str(tmp_path / "final"), "--eval-set", str(tmp_path / "eval.npz"),
                   "--out", str(tmp_path / "reports"), "--name", "cli", "--device", "cpu",
                   "--sigmas", "0,0.1", "--factors", "2", "--time-factors", "2",
                   "--batch-size", "4", "--num-pairs", "2"])
    assert [r["sigma"] for r in result["noise"]] == [0.0, 0.1]
    assert (tmp_path / "reports" / "cli.json").exists()
    assert (tmp_path / "reports" / "cli.md").exists()
