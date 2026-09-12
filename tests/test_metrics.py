import json

import numpy as np
import pytest

from vqvae_latent_actions.training.metrics import MetricAccumulator, masked_errors, write_report


def test_masked_errors_ignore_padding():
    actions = np.zeros((2, 3, 4))
    recon = np.ones((2, 3, 4)) * 0.5
    dims = np.array([[True, True, False, False], [True, False, False, False]])
    steps = np.array([[True, True, True], [True, True, False]])
    valid = steps[:, :, None] & dims[:, None, :]
    errors = masked_errors(actions, recon, valid)
    np.testing.assert_allclose(errors["mse"], [0.25, 0.25])
    np.testing.assert_allclose(errors["l1"], [0.5, 0.5])
    recon[0, :, 2] = 100.0                       # padded dimension is ignored
    assert masked_errors(actions, recon, valid)["max_abs"][0] == 0.5
    recon[1, 2, 0] = 100.0                       # padded time step is ignored
    assert masked_errors(actions, recon, valid)["max_abs"][1] == 0.5


def test_accumulator_summary_and_report(tmp_path):
    acc = MetricAccumulator(["e0", "e1"])
    actions = np.zeros((4, 2, 3))
    recon = actions + 0.1
    valid = np.ones((4, 2, 3), bool)
    acc.add(actions, recon, valid, [0, 0, 1, 1], [5, 7, 9, 9], failed=[False, True, False, False])
    summary = acc.summary()
    assert summary["total"]["n"] == 4 and summary["total"]["decode_failures"] == 1
    assert summary["total"]["l1"] == pytest.approx(0.1)
    assert summary["per_embodiment"]["e0"]["tokens_mean"] == 6.0
    write_report(tmp_path / "r.json", "unit", summary, {"note": 1})
    payload = json.loads((tmp_path / "r.json").read_text())
    assert payload["arm"] == "unit" and payload["note"] == 1
    assert "| e0 |" in (tmp_path / "r.md").read_text()


def test_matches_tokenizer_arms_metrics():
    """The numbers must stay comparable with the FAST/BEAST/ActionCodec/OAT table."""
    reference = pytest.importorskip("tokenizer_arms.metrics")
    rng = np.random.default_rng(0)
    actions = rng.normal(size=(6, 4, 5))
    recon = actions + rng.normal(scale=0.1, size=actions.shape)
    valid = rng.random((6, 4, 5)) > 0.3
    valid[:, :, 0] = True
    ours = MetricAccumulator(["a", "b"])
    theirs = reference.MetricAccumulator(["a", "b"])
    emb, tokens = [0, 0, 0, 1, 1, 1], [10] * 6
    ours.add(actions, recon, valid, emb, tokens)
    theirs.add(actions, recon, valid, emb, tokens)
    assert ours.summary() == theirs.summary()
