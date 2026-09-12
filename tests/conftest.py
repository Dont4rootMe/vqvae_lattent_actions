"""Shared fixtures. Synthetic data comes from action_chunks' own test helpers, so no cluster access is needed."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
ACTION_CHUNKS = Path(os.environ.get("ACTION_CHUNKS_REPO", ROOT.parent / "action_chunks"))
LAYOUT_YAML = ROOT / "configs" / "layout" / "bimanual_rotation6d.yaml"

for _p in (ROOT, ACTION_CHUNKS):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


@pytest.fixture(scope="session")
def layout():
    from vqvae_latent_actions.layout import UnifiedLayout
    return UnifiedLayout.from_yaml(LAYOUT_YAML)


@pytest.fixture(scope="session")
def action_chunks_helpers():
    """action_chunks/tests/helpers.py loaded by path (it is not an installed package)."""
    path = ACTION_CHUNKS / "tests" / "helpers.py"
    if not path.exists():
        pytest.skip(f"action_chunks test helpers not found at {path}")
    spec = importlib.util.spec_from_file_location("ac_test_helpers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def tiny_manifest(tmp_path_factory, action_chunks_helpers):
    """Two joint embodiments (2 real dims each inside the 119-dim layout), a handful of short episodes."""
    from action_chunks.manifest import Manifest
    h = action_chunks_helpers
    root = tmp_path_factory.mktemp("data")
    rng = np.random.default_rng(0)
    eps_a = [np.cumsum(rng.normal(0, 0.05, size=(n, 2)), axis=0) for n in (80, 64, 120, 96, 70, 90)]
    eps_b = [np.cumsum(rng.normal(0, 0.05, size=(n, 2)), axis=0) for n in (100, 60, 75, 88)]
    ra = h.write_lerobot_dataset(root / "a", eps_a)
    rb = h.write_lerobot_dataset(root / "b", eps_b, validation_list=[1])
    m = Manifest(source={}, layout=h.LAYOUT, total_dim=h.TOTAL_DIM,
                 embodiments=[h.joint_spec(ra, index=0, weight=2.0),
                              h.joint_spec(rb, index=1, weight=1.0, validation=True)])
    out = tmp_path_factory.mktemp("manifest")
    m.save(out)
    return out


@pytest.fixture(scope="session")
def tiny_eval_set(tiny_manifest, tmp_path_factory):
    from vqvae_latent_actions.data.chunks import build_eval_set
    out = tmp_path_factory.mktemp("eval") / "eval.npz"
    return build_eval_set(tiny_manifest, None, out, per_embodiment=6, split="all", num_workers=0)


@pytest.fixture
def tiny_model_config(layout):
    """Small but structurally complete model: FSQ [4,4] -> V=16, N=4 tokens."""
    from vqvae_latent_actions.models.hier_tokenizer import HierTokenizerConfig
    return HierTokenizerConfig(
        layout=layout.to_dict(), horizon=10, max_horizon=16, num_tokens=4, dim=32, heads=4, free_queries=2,
        enc_step_layers=1, enc_time_layers=1, enc_latent_layers=1, dec_latent_layers=1, dec_layers=1,
        quantizer={"type": "fsq", "levels": [4, 4], "commitment": 0.25},
    )


@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")
