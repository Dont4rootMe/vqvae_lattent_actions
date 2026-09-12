# Hierarchical action tokenizer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a learned discrete tokenizer that encodes a `T x 119` unified-action-space chunk with its `{0,1}^{T x 119}` padding mask into `N` tokens from a vocabulary of `V <= 2048`, trains on the R0_v2.1 mixture through `action_chunks`, and is scored on the shared eval set next to FAST/BEAST/ActionCodec/OAT.

**Architecture:** Pointwise embedding (learned `[PAD]`, per-dim, per-group and time embeddings) → per-step cross-attention by `K = 17 groups + free` queries with group-restricted attention masks → self-attention over `T*K` tokens (dynamics) → Perceiver latents (`N`) → quantizer (FSQ default) → mirrored decoder conditioned on the same mask → per-dim heads, zeros in padding, masked MSE.

**Tech Stack:** PyTorch 2.7+ (SDPA), accelerate (DDP, bf16), hydra + omegaconf (configs), Comet ML (experiment tracking, online or offline), numpy, pytest. Data comes from the `action_chunks` package (external dependency, not vendored).

## Global Constraints

- Unified space `bimanual_rotation6d`: 119 dims, 17 groups, offsets exactly as in `configs/layout/bimanual_rotation6d.yaml`; layout is never hardcoded in model code.
- `T` (horizon), `N` (tokens per chunk), `V` (vocabulary) are independent config values; `V <= 2048` (not more distinct tokens than FAST); report `N * log2(V)` bits with every result.
- Only the action batch and its mask enter the model. No robot state, no images, no embodiment id.
- Mask is `{0,1}^{T x 119}` (arbitrary pattern), given to encoder and decoder; decoder output is exactly 0 at masked positions; loss and metrics count only unmasked positions.
- Metrics identical to `tokenizer_arms.metrics` (RMSE, L1, max-abs over valid entries; totals and per embodiment) so rows go straight into the four-arm table.
- Experiment tracking: Comet ML only (no tensorboard, no ClearML). API key is never committed: it lives in `~/.comet.config` locally and `/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/.comet.config` on the cluster; code reads `COMET_API_KEY`/`COMET_CONFIG` from the environment. Workspace `dont4rootme`, project `hier-action-tokenizer`.
- Python 3.11, torch >= 2.7 (cluster env `ai_lerobot_qwen3vl_develop`); `comet_ml` on the cluster comes from `$BASE/pylib_comet` via `PYTHONPATH` (PyPI is unreachable from the IB node).
- TDD: every task writes a failing test first, then the implementation, then commits. `pytest -q` must be green at the end of each task.
- Local test venv: `/Users/artemon/projects/action_chunks/.venv` (torch 2.14, accelerate 1.14, hydra 1.3.6); `PYTHONPATH` must include the repo and `/Users/artemon/projects/action_chunks`.

## File Structure

```
vqvae_latent_actions/
  layout.py                 UnifiedLayout: groups -> dim ranges, group index, membership mask, dim weights   [T1]
  models/quantizers.py      Quantizer interface + FSQ, VQEMA, LFQ, Gumbel + build_quantizer + code_usage     [T2]
  models/blocks.py          Attention, CrossBlock, SelfBlock, FFBlock (pre-LN, SDPA, bool masks)             [T3]
  models/hier_tokenizer.py  HierTokenizerConfig + HierActionTokenizer (encoder/decoder/loss/save/load)       [T3]
  data/chunks.py            batch_to_inputs, train_loader, EvalSet (+build/save/load), layout_from_manifest  [T4]
  training/metrics.py       masked_errors, MetricAccumulator, write_report (port of tokenizer_arms.metrics)  [T5]
  training/evaluate.py      evaluate_tokenizer, padding_invariance_mismatch, model_report_extra              [T5]
  training/comet.py         RunLogger: online/offline/disabled Comet experiment + jsonl mirror               [T6]
  training/loop.py          TrainConfig + train(): accelerate DDP, resume, eval, checkpoints, final export   [T6]
train.py                    hydra entry point                                                                [T7]
configs/                    config.yaml, model/hier_{fsq,vq,lfq,gumbel}.yaml, data/r0_v2_1.yaml, layout/     [T1,T7]
launchers/                  common.sh, train.sh, smoke.sh, submit.sh, comet_upload.sh                        [T7]
tests/                      conftest.py + test_{layout,quantizers,model,data,metrics,evaluate,loop,entry}.py [T1-T7]
docs/superpowers/           spec, this plan, runs.md journal                                                 [T7]
```

Deleted on the way: `dataset.py`, `config.ckpt`, `lfq_vqvae.txt`, `train.sh` (old), `vqvae_latent_actions/models/{fsq,lfq,gmb}_vqvae.py`, `vqvae_latent_actions/utils/`, `vqvae_latent_actions/training/{loop,eval}.py` (rewritten).

---

### Task 1: Scaffold, layout, test harness

**Files:**
- Create: `vqvae_latent_actions/layout.py`, `tests/conftest.py`, `tests/test_layout.py`, `configs/layout/bimanual_rotation6d.yaml` (already copied)
- Modify: `pyproject.toml`, `.gitignore`
- Delete: `dataset.py`, `config.ckpt`, `lfq_vqvae.txt`

**Interfaces:**
- Produces: `UnifiedLayout` with `.names/.starts/.ends`, `.total_dim`, `.num_groups`, `.intervals()`, `.group_index() -> LongTensor[D]`, `.membership() -> BoolTensor[G,D]`, `.dim_weights(dict|None) -> FloatTensor[D]`, `.to_dict()/.from_dict()`, `.from_intervals()`, `.from_manifest()`, `.from_yaml()`.
- Produces (tests): fixtures `layout`, `tiny_layout`, `action_chunks_helpers`, `tiny_manifest`, `tiny_eval_set`, `device`.

- [ ] **Step 1: Write the failing test**

```python
# file: tests/test_layout.py
import pytest
import torch

from vqvae_latent_actions.layout import UnifiedLayout


def test_from_yaml_matches_the_fork_layout(layout):
    assert layout.total_dim == 119 and layout.num_groups == 17
    iv = layout.intervals()
    assert iv["left_arm.joints"] == (0, 8)
    assert iv["left_arm.cartesian_position"] == (8, 11)
    assert iv["left_arm.cartesian_rotation"] == (11, 17)      # rotation_6d -> 6 dims
    assert iv["left_arm.gripper"] == (17, 18)
    assert iv["left_hand.joints"] == (18, 42)
    assert iv["right_arm.joints"] == (42, 50)
    assert iv["right_arm.gripper"] == (59, 60)
    assert iv["right_hand.joints"] == (60, 84)
    assert iv["head.joints"] == (84, 87)
    assert iv["torso.joints"] == (87, 92)
    assert iv["base.velocity"] == (92, 98)
    assert iv["base.wheel_phase"] == (98, 104)
    assert iv["left_leg.joints"] == (104, 110)
    assert iv["right_leg.joints"] == (110, 116)
    assert iv["base.wheel_joints"] == (116, 119)


def test_group_index_and_membership(layout):
    gi = layout.group_index()
    m = layout.membership()
    assert gi.shape == (119,) and gi.dtype == torch.long
    assert int(gi[0]) == 0 and int(gi[18]) == 4 and int(gi[118]) == 16
    assert m.shape == (17, 119) and m.dtype == torch.bool
    assert int(m.sum()) == 119                                  # every dim belongs to exactly one group
    assert int(m.sum(dim=1)[4]) == 24                            # left_hand.joints
    assert bool(m[gi, torch.arange(119)].all())                  # membership agrees with group_index


def test_dim_weights(layout):
    w = layout.dim_weights({"left_arm.gripper": 5.0, "base.velocity": 2.0})
    assert w.shape == (119,)
    assert float(w[17]) == 5.0 and float(w[92]) == 2.0 and float(w[0]) == 1.0
    assert torch.equal(layout.dim_weights(None), torch.ones(119))


def test_from_intervals_rejects_gaps_and_overlaps():
    with pytest.raises(ValueError):
        UnifiedLayout.from_intervals({"a": (0, 2), "b": (3, 5)})
    with pytest.raises(ValueError):
        UnifiedLayout.from_intervals({"a": (0, 3), "b": (2, 5)})
    with pytest.raises(ValueError):
        UnifiedLayout.from_intervals({"a": (0, 0)})


def test_dict_roundtrip(layout):
    again = UnifiedLayout.from_dict(layout.to_dict())
    assert again == layout and again.intervals() == layout.intervals()
```

```python
# file: tests/conftest.py
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/artemon/projects/vqvae_lattent_actions && PYTHONPATH=.:/Users/artemon/projects/action_chunks /Users/artemon/projects/action_chunks/.venv/bin/pytest -q tests/test_layout.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'vqvae_latent_actions.layout'`

- [ ] **Step 3: Write the implementation**

```python
# file: vqvae_latent_actions/layout.py
"""Unified action-space layout: named semantic groups over the flat action vector.

The layout is data, never hardcoded in the model: it comes from the fork's yaml
(`configs/layout/bimanual_rotation6d.yaml`), from an `action_chunks` manifest, or from a dict
stored inside a model checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

ROTATION_DIMS = {"rotation_6d": 6, "quaternion": 4, "euler": 3, "axis_angle": 3, "rotation_matrix": 9}


@dataclass(frozen=True)
class UnifiedLayout:
    """Contiguous named groups tiling `[0, total_dim)` in order."""

    names: tuple[str, ...]
    starts: tuple[int, ...]
    ends: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.names or not (len(self.names) == len(self.starts) == len(self.ends)):
            raise ValueError("names, starts and ends must be non-empty and of equal length")
        position = 0
        for name, start, end in zip(self.names, self.starts, self.ends):
            if int(start) != position or int(end) <= int(start):
                raise ValueError(f"groups must tile [0, total_dim) without gaps or overlaps; {name!r} breaks at {start}")
            position = int(end)

    @property
    def total_dim(self) -> int:
        return int(self.ends[-1])

    @property
    def num_groups(self) -> int:
        return len(self.names)

    def intervals(self) -> dict[str, tuple[int, int]]:
        return {n: (int(s), int(e)) for n, s, e in zip(self.names, self.starts, self.ends)}

    def group_index(self) -> torch.Tensor:
        """[D] long: group id of every dimension."""
        out = torch.zeros(self.total_dim, dtype=torch.long)
        for g, (s, e) in enumerate(zip(self.starts, self.ends)):
            out[int(s):int(e)] = g
        return out

    def membership(self) -> torch.Tensor:
        """[G, D] bool: True where the dimension belongs to the group."""
        out = torch.zeros(self.num_groups, self.total_dim, dtype=torch.bool)
        for g, (s, e) in enumerate(zip(self.starts, self.ends)):
            out[g, int(s):int(e)] = True
        return out

    def dim_weights(self, group_weights: Mapping[str, float] | None) -> torch.Tensor:
        """[D] float: per-dimension loss weight, 1.0 unless the group is listed."""
        out = torch.ones(self.total_dim, dtype=torch.float32)
        for name, weight in (group_weights or {}).items():
            if name not in self.names:
                raise KeyError(f"unknown group {name!r}; known: {list(self.names)}")
            s, e = self.intervals()[name]
            out[s:e] = float(weight)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"names": list(self.names), "starts": [int(s) for s in self.starts], "ends": [int(e) for e in self.ends]}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UnifiedLayout":
        return cls(tuple(payload["names"]), tuple(int(s) for s in payload["starts"]), tuple(int(e) for e in payload["ends"]))

    @classmethod
    def from_intervals(cls, intervals: Mapping[str, tuple[int, int]]) -> "UnifiedLayout":
        items = sorted(intervals.items(), key=lambda kv: kv[1][0])
        return cls(tuple(k for k, _ in items), tuple(int(v[0]) for _, v in items), tuple(int(v[1]) for _, v in items))

    @classmethod
    def from_manifest(cls, manifest: Any) -> "UnifiedLayout":
        """`manifest` is an action_chunks Manifest (or anything with `layout_intervals()`)."""
        return cls.from_intervals(manifest.layout_intervals())

    @classmethod
    def from_yaml(cls, path: str | Path, rotation_representation: str | None = None) -> "UnifiedLayout":
        payload = yaml.safe_load(Path(path).read_text())
        rotation = rotation_representation or payload.get("rotation_representation", "rotation_6d")
        if rotation not in ROTATION_DIMS:
            raise ValueError(f"unknown rotation representation {rotation!r}")
        names, starts, ends, position = [], [], [], 0
        for name, group in payload["groups"].items():
            kind = group.get("kind", "vector")
            if kind == "rotation":
                dim = ROTATION_DIMS[rotation]
            elif kind == "vector":
                dim = int(group["dim"] if "dim" in group else group["max_dim"])
            else:
                raise ValueError(f"unknown group kind {kind!r} for {name!r}")
            names.append(name); starts.append(position); ends.append(position + dim); position += dim
        return cls(tuple(names), tuple(starts), tuple(ends))


__all__ = ["UnifiedLayout", "ROTATION_DIMS"]
```

```toml
# file: pyproject.toml
[build-system]
requires = ["setuptools>=65", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "vqvae-latent-actions"
version = "0.2.0"
description = "Hierarchical discrete tokenizer for VLA action chunks in the unified action space"
authors = [{name = "SberRobotics"}]
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
  "torch>=2.4",
  "numpy",
  "pyyaml",
  "accelerate",
  "hydra-core",
  "omegaconf",
]

[project.optional-dependencies]
tracking = ["comet_ml"]
lfq = ["vector-quantize-pytorch"]
dev = ["pytest", "ruff"]

[tool.setuptools.packages.find]
include = ["vqvae_latent_actions*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
filterwarnings = ["ignore::DeprecationWarning"]

[tool.ruff]
line-length = 120
```

```gitignore
# file: .gitignore
__pycache__
*.pyc
*.pyo
.pytest_cache/
.ruff_cache/

outputs/
runs/
wandb/
.cometml-runs/
comet_offline/
*.zip

# credentials must never be committed
.comet.config
*.api_key
COMET_API_KEY

.DS_Store
.env
.env.*
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONPATH=.:/Users/artemon/projects/action_chunks /Users/artemon/projects/action_chunks/.venv/bin/pytest -q tests/test_layout.py`
Expected: 5 passed

- [ ] **Step 5: Delete the superseded files and commit**

```bash
git rm -q dataset.py config.ckpt lfq_vqvae.txt
git add -A
git commit -m "layout: unified action space groups; repo scaffold for the hierarchical tokenizer"
```

---

### Task 2: Quantizers behind one interface

**Files:**
- Create: `vqvae_latent_actions/models/quantizers.py`, `tests/test_quantizers.py`
- Modify: `vqvae_latent_actions/models/__init__.py`
- Delete: `vqvae_latent_actions/models/{fsq_vqvae,lfq_vqvae,gmb_vqvae}.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `QuantizerOutput(codes, indices, aux_loss)`; `Quantizer` with attributes `vocab_size`, `code_dim`, `out_dim`, methods `forward(z[B,N,code_dim]) -> QuantizerOutput`, `indices_to_codes(idx) -> [.., out_dim]`, property `bits_per_token`; classes `FSQ`, `VQEMA`, `LFQWrapper`, `GumbelQuantizer`; `build_quantizer(cfg: dict) -> Quantizer`; `code_usage(indices, vocab_size) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# file: tests/test_quantizers.py
import math

import pytest
import torch

from vqvae_latent_actions.models.quantizers import (FSQ, GumbelQuantizer, VQEMA, build_quantizer, code_usage)


def test_fsq_vocabulary_grid_and_roundtrip():
    q = FSQ(levels=[8, 8, 8, 4])
    assert q.vocab_size == 2048 and q.code_dim == 4 and q.out_dim == 4
    assert q.bits_per_token == pytest.approx(11.0)
    z = torch.randn(3, 5, 4) * 3
    out = q(z)
    assert out.codes.shape == z.shape and out.indices.shape == (3, 5)
    assert int(out.indices.min()) >= 0 and int(out.indices.max()) < q.vocab_size
    torch.testing.assert_close(q.indices_to_codes(out.indices), out.codes)          # index <-> code agree
    grid = q.indices_to_codes(torch.arange(q.vocab_size))                            # every code is distinct
    assert grid.shape == (q.vocab_size, 4)
    assert len({tuple(row.tolist()) for row in grid}) == q.vocab_size
    saturated = q(torch.randn(16, 32, 4) * 50)                                       # extreme latents stay in range
    assert int(saturated.indices.min()) >= 0 and int(saturated.indices.max()) < q.vocab_size


def test_fsq_straight_through_gradient_and_commitment():
    q = FSQ(levels=[4, 4], commitment=0.25)
    z = torch.randn(2, 3, 2, requires_grad=True)
    out = q(z)
    (out.codes.sum() + out.aux_loss).backward()
    assert z.grad is not None and torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0
    assert float(out.aux_loss) >= 0.0


def test_vqema_learns_clusters_and_reports_indices():
    torch.manual_seed(0)
    q = VQEMA(vocab_size=8, code_dim=3, decay=0.8)
    centers = torch.randn(8, 3) * 4
    def batch():
        idx = torch.randint(0, 8, (64,))
        return (centers[idx] + torch.randn(64, 3) * 0.05).view(4, 16, 3)
    q.train()
    start = batch()
    first = float(((q(start).codes - start) ** 2).mean())
    for _ in range(40):
        q(batch())
    errors = []
    for _ in range(5):
        sample = batch()
        errors.append(((q(sample).codes - sample) ** 2).mean())
    err = float(torch.stack(errors).mean())
    assert err < first and err < 0.5                                   # codebook moved onto the clusters
    out = q(batch())
    assert out.indices.shape == (4, 16) and int(out.indices.max()) < 8
    torch.testing.assert_close(q.indices_to_codes(out.indices), q.codebook[out.indices])
    q.eval()
    before = q.codebook.clone()
    q(batch())
    assert torch.equal(before, q.codebook)                             # eval mode never updates the codebook


def test_gumbel_is_hard_and_deterministic_in_eval():
    q = GumbelQuantizer(vocab_size=16, code_dim=8, out_dim=8, temperature=1.0)
    z = torch.randn(2, 4, 8, requires_grad=True)
    q.train()
    out = q(z)
    assert out.codes.shape == (2, 4, 8) and out.indices.shape == (2, 4)
    out.codes.sum().backward()
    assert z.grad is not None and z.grad.abs().sum() > 0
    q.eval()
    a, b = q(z.detach()), q(z.detach())
    assert torch.equal(a.indices, b.indices)
    torch.testing.assert_close(a.codes, q.indices_to_codes(a.indices))


def test_build_quantizer_and_usage():
    q = build_quantizer({"type": "fsq", "levels": [4, 4, 4]})
    assert isinstance(q, FSQ) and q.vocab_size == 64
    assert isinstance(build_quantizer({"type": "vq", "vocab_size": 32, "code_dim": 4}), VQEMA)
    with pytest.raises(ValueError):
        build_quantizer({"type": "nope"})
    usage = code_usage(torch.tensor([[0, 1], [1, 1]]), vocab_size=4)
    assert usage["codes_used"] == 2 and usage["usage_percent"] == pytest.approx(50.0)
    assert usage["perplexity"] == pytest.approx(math.exp(-(0.25 * math.log(0.25) + 0.75 * math.log(0.75))), rel=1e-5)


def test_lfq_optional():
    pytest.importorskip("vector_quantize_pytorch")
    q = build_quantizer({"type": "lfq", "vocab_size": 256})
    z = torch.randn(2, 3, q.code_dim)
    out = q(z)
    assert q.vocab_size == 256 and q.code_dim == 8
    assert out.indices.shape == (2, 3) and int(out.indices.max()) < 256
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=.:/Users/artemon/projects/action_chunks /Users/artemon/projects/action_chunks/.venv/bin/pytest -q tests/test_quantizers.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'vqvae_latent_actions.models.quantizers'`

- [ ] **Step 3: Write the implementation**

```python
# file: vqvae_latent_actions/models/quantizers.py
"""Quantizers: N continuous latents -> N discrete tokens, behind one interface.

FSQ is the default (no codebook collapse, deterministic, vocabulary = product of levels). VQ-EMA covers
arbitrary vocabulary sizes, LFQ and Gumbel-softmax are kept from the earlier experiments in this repo.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class QuantizerOutput:
    codes: Tensor      # [B, N, out_dim], quantized values with a straight-through gradient
    indices: Tensor    # [B, N] long in [0, vocab_size)
    aux_loss: Tensor   # scalar, added to the reconstruction loss


class Quantizer(nn.Module):
    """Interface. `code_dim` is what the encoder must produce, `out_dim` what the decoder consumes."""

    vocab_size: int
    code_dim: int
    out_dim: int

    @property
    def bits_per_token(self) -> float:
        return math.log2(self.vocab_size)

    def forward(self, z: Tensor) -> QuantizerOutput:  # pragma: no cover - interface
        raise NotImplementedError

    def indices_to_codes(self, indices: Tensor) -> Tensor:  # pragma: no cover - interface
        raise NotImplementedError


def round_ste(z: Tensor) -> Tensor:
    return z + (z.round() - z).detach()


class FSQ(Quantizer):
    """Finite scalar quantization (arXiv 2309.15505): each latent dim is rounded onto its own grid."""

    def __init__(self, levels: Sequence[int], commitment: float = 0.25, eps: float = 1e-3) -> None:
        super().__init__()
        levels = [int(x) for x in levels]
        if not levels or any(x < 2 for x in levels):
            raise ValueError(f"levels must be a non-empty sequence of ints >= 2, got {levels}")
        lv = torch.tensor(levels, dtype=torch.float32)
        half = (lv - 1) * (1 + eps) / 2
        offset = ((torch.tensor(levels) % 2) == 0).to(torch.float32) * 0.5
        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.long), persistent=False)
        self.register_buffer("_half", half, persistent=False)
        self.register_buffer("_offset", offset, persistent=False)
        self.register_buffer("_shift", torch.atanh(offset / half), persistent=False)
        self.register_buffer("_half_width", (torch.tensor(levels) // 2).to(torch.float32), persistent=False)
        basis = torch.cumprod(torch.tensor([1] + levels[:-1], dtype=torch.long), dim=0)
        self.register_buffer("_basis", basis, persistent=False)
        self.levels = levels
        self.vocab_size = int(math.prod(levels))
        self.code_dim = self.out_dim = len(levels)
        self.commitment = float(commitment)

    def _bound(self, z: Tensor) -> Tensor:
        return torch.tanh(z + self._shift) * self._half - self._offset

    def forward(self, z: Tensor) -> QuantizerOutput:
        z = z.float()
        bounded = self._bound(z)
        grid = round_ste(bounded)
        codes = grid / self._half_width
        with torch.no_grad():
            shifted = (grid + (self._levels // 2)).round().long().clamp_(min=0)
            shifted = torch.minimum(shifted, self._levels - 1)
            indices = (shifted * self._basis).sum(dim=-1)
        aux = self.commitment * F.mse_loss(bounded, grid.detach()) if self.commitment else z.new_zeros(())
        return QuantizerOutput(codes=codes, indices=indices, aux_loss=aux)

    def indices_to_codes(self, indices: Tensor) -> Tensor:
        digits = torch.div(indices.unsqueeze(-1), self._basis, rounding_mode="floor") % self._levels
        return (digits - (self._levels // 2)).float() / self._half_width


class VQEMA(Quantizer):
    """Classic VQ with EMA codebook updates, dead-code replacement and a commitment loss."""

    def __init__(self, vocab_size: int, code_dim: int, commitment: float = 0.25, decay: float = 0.99,
                 eps: float = 1e-5, dead_threshold: float = 1.0) -> None:
        super().__init__()
        self.vocab_size, self.code_dim, self.out_dim = int(vocab_size), int(code_dim), int(code_dim)
        self.commitment, self.decay, self.eps, self.dead_threshold = float(commitment), float(decay), float(eps), float(dead_threshold)
        codebook = torch.randn(self.vocab_size, self.code_dim) * 0.1
        self.register_buffer("codebook", codebook)
        self.register_buffer("cluster_size", torch.ones(self.vocab_size))
        self.register_buffer("embed_sum", codebook.clone())
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))

    @staticmethod
    def _sync(tensor: Tensor, average: bool = False) -> Tensor:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(tensor)
            if average:
                tensor /= torch.distributed.get_world_size()
        return tensor

    @staticmethod
    def _broadcast(tensor: Tensor) -> Tensor:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(tensor, src=0)
        return tensor

    @torch.no_grad()
    def _initialize(self, flat: Tensor) -> None:
        pick = torch.randint(0, flat.shape[0], (self.vocab_size,), device=flat.device)
        codes = self._broadcast(flat[pick].clone())
        self.codebook.copy_(codes)
        self.embed_sum.copy_(codes)
        self.cluster_size.fill_(1.0)
        self.initialized.fill_(True)

    def forward(self, z: Tensor) -> QuantizerOutput:
        z = z.float()
        flat = z.reshape(-1, self.code_dim)
        if self.training and not bool(self.initialized):
            self._initialize(flat.detach())
        distances = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ self.codebook.t() + self.codebook.pow(2).sum(1)
        indices = distances.argmin(dim=1)
        quantized = self.codebook[indices]
        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(indices, self.vocab_size).to(flat.dtype)
                counts = self._sync(onehot.sum(0))
                sums = self._sync(onehot.t() @ flat.detach())
                self.cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
                self.embed_sum.mul_(self.decay).add_(sums, alpha=1 - self.decay)
                total = self.cluster_size.sum()
                smoothed = (self.cluster_size + self.eps) / (total + self.vocab_size * self.eps) * total
                self.codebook.copy_(self.embed_sum / smoothed.unsqueeze(1))
                dead = self.cluster_size < self.dead_threshold
                if bool(dead.any()):
                    pick = torch.randint(0, flat.shape[0], (int(dead.sum()),), device=flat.device)
                    replacement = self._broadcast(flat.detach()[pick].clone())
                    self.codebook[dead] = replacement
                    self.embed_sum[dead] = replacement
                    self.cluster_size[dead] = 1.0
        aux = self.commitment * F.mse_loss(flat, quantized.detach())
        codes = flat + (quantized - flat).detach()
        return QuantizerOutput(codes=codes.view_as(z), indices=indices.view(z.shape[:-1]), aux_loss=aux)

    def indices_to_codes(self, indices: Tensor) -> Tensor:
        return self.codebook[indices]


class LFQWrapper(Quantizer):
    """Lookup-free quantization from vector_quantize_pytorch (optional dependency)."""

    def __init__(self, vocab_size: int, entropy_loss_weight: float = 0.1, diversity_gamma: float = 1.0) -> None:
        super().__init__()
        from vector_quantize_pytorch import LFQ

        bits = int(math.log2(vocab_size))
        if 2 ** bits != int(vocab_size):
            raise ValueError(f"LFQ vocabulary must be a power of two, got {vocab_size}")
        self.lfq = LFQ(codebook_size=int(vocab_size), dim=bits, num_codebooks=1,
                       entropy_loss_weight=float(entropy_loss_weight), diversity_gamma=float(diversity_gamma))
        self.vocab_size = int(vocab_size)
        self.code_dim = self.out_dim = bits

    def forward(self, z: Tensor) -> QuantizerOutput:
        codes, indices, aux = self.lfq(z.float())
        return QuantizerOutput(codes=codes, indices=indices.reshape(z.shape[:-1]).long(), aux_loss=aux)

    def indices_to_codes(self, indices: Tensor) -> Tensor:
        return self.lfq.indices_to_codes(indices)


class GumbelQuantizer(Quantizer):
    """Categorical latent with the Gumbel-softmax straight-through estimator and a learned embedding table."""

    def __init__(self, vocab_size: int, code_dim: int, out_dim: int | None = None, temperature: float = 1.0,
                 entropy_weight: float = 0.0) -> None:
        super().__init__()
        self.vocab_size, self.code_dim = int(vocab_size), int(code_dim)
        self.out_dim = int(out_dim or code_dim)
        self.temperature, self.entropy_weight = float(temperature), float(entropy_weight)
        self.to_logits = nn.Linear(self.code_dim, self.vocab_size)
        self.embedding = nn.Embedding(self.vocab_size, self.out_dim)

    def set_temperature(self, value: float) -> None:
        self.temperature = float(value)

    def forward(self, z: Tensor) -> QuantizerOutput:
        logits = self.to_logits(z.float())
        if self.training:
            soft = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)
            indices = soft.argmax(dim=-1)
            codes = soft @ self.embedding.weight
        else:
            indices = logits.argmax(dim=-1)
            codes = self.embedding(indices)
        aux = z.new_zeros(())
        if self.entropy_weight:
            average = logits.softmax(dim=-1).mean(dim=tuple(range(logits.dim() - 1)))
            aux = self.entropy_weight * (average * (average + 1e-9).log()).sum()   # maximise marginal entropy
        return QuantizerOutput(codes=codes, indices=indices, aux_loss=aux)

    def indices_to_codes(self, indices: Tensor) -> Tensor:
        return self.embedding(indices)


def build_quantizer(config: Mapping[str, Any]) -> Quantizer:
    config = dict(config)
    kind = str(config.pop("type", "fsq")).lower()
    if kind == "fsq":
        return FSQ(**config)
    if kind in ("vq", "vqema", "vq_ema"):
        return VQEMA(**config)
    if kind == "lfq":
        return LFQWrapper(**config)
    if kind == "gumbel":
        return GumbelQuantizer(**config)
    raise ValueError(f"unknown quantizer type {kind!r}")


def code_usage(indices: Tensor, vocab_size: int) -> dict[str, float]:
    """Codebook usage of a batch of token indices: how many codes appear and how evenly."""
    counts = torch.bincount(indices.reshape(-1).long(), minlength=int(vocab_size)).float()
    probs = counts / counts.sum().clamp_min(1.0)
    nonzero = probs[probs > 0]
    entropy = float(-(nonzero * nonzero.log()).sum())
    used = int((counts > 0).sum())
    return {"codes_used": used, "usage_percent": 100.0 * used / float(vocab_size),
            "perplexity": float(math.exp(entropy)), "entropy_bits": entropy / math.log(2)}


__all__ = ["Quantizer", "QuantizerOutput", "FSQ", "VQEMA", "LFQWrapper", "GumbelQuantizer",
           "build_quantizer", "code_usage", "round_ste"]
```

```python
# file: vqvae_latent_actions/models/__init__.py
"""Model registry."""
from .hier_tokenizer import HierActionTokenizer, HierTokenizerConfig
from .quantizers import (FSQ, GumbelQuantizer, LFQWrapper, Quantizer, QuantizerOutput, VQEMA, build_quantizer,
                         code_usage)

__all__ = ["HierActionTokenizer", "HierTokenizerConfig", "Quantizer", "QuantizerOutput", "FSQ", "VQEMA",
           "LFQWrapper", "GumbelQuantizer", "build_quantizer", "code_usage"]
```

Note: `models/__init__.py` imports `hier_tokenizer`, which lands in Task 3. Until then keep the quantizer imports only; the final content above is committed at the end of Task 3.

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONPATH=.:/Users/artemon/projects/action_chunks /Users/artemon/projects/action_chunks/.venv/bin/pytest -q tests/test_quantizers.py`
Expected: 6 passed

- [ ] **Step 5: Delete the superseded models and commit**

```bash
git rm -q vqvae_latent_actions/models/fsq_vqvae.py vqvae_latent_actions/models/lfq_vqvae.py vqvae_latent_actions/models/gmb_vqvae.py
git add -A
git commit -m "quantizers: FSQ, VQ-EMA, LFQ and Gumbel behind one interface"
```

---

### Task 3: Attention blocks and the hierarchical tokenizer

**Files:**
- Create: `vqvae_latent_actions/models/blocks.py`, `vqvae_latent_actions/models/hier_tokenizer.py`, `tests/test_model.py`
- Modify: `vqvae_latent_actions/models/__init__.py` (final content from Task 2)

**Interfaces:**
- Consumes: `UnifiedLayout` (Task 1), `build_quantizer`, `QuantizerOutput`, `code_usage` (Task 2).
- Produces: `HierTokenizerConfig` (dataclass, `to_dict`/`from_dict`), `HierActionTokenizer` with
  `encode_step_tokens(actions, mask) -> [B,T,K,d]`, `encode_continuous(actions, mask) -> [B,N,code_dim]`,
  `quantize(z) -> QuantizerOutput`, `decode_latents(codes, mask) -> [B,T,D]`, `forward(actions, mask) -> dict`,
  `tokenize(actions, mask) -> [B,N] long`, `detokenize(tokens, mask) -> [B,T,D]`,
  `encode_decode(actions, mask) -> (tokens, recon)`, `save_pretrained(dir)`, `from_pretrained(dir)`,
  properties `num_tokens`, `vocab_size`, `bits_per_chunk`.

- [ ] **Step 1: Write the failing test**

```python
# file: tests/test_model.py
import math

import pytest
import torch

from vqvae_latent_actions.models.hier_tokenizer import HierActionTokenizer, HierTokenizerConfig


def _mask(layout, batch: int, horizon: int, groups: list[str], time_cut: int | None = None) -> torch.Tensor:
    m = torch.zeros(batch, horizon, layout.total_dim, dtype=torch.bool)
    for name in groups:
        s, e = layout.intervals()[name]
        m[:, :, s:e] = True
    if time_cut is not None:
        m[:, time_cut:, :] = False
    return m


@pytest.fixture
def model(tiny_model_config):
    torch.manual_seed(0)
    return HierActionTokenizer(tiny_model_config).eval()


def test_shapes_tokens_and_budget(model, layout):
    b, t, d = 3, 10, layout.total_dim
    mask = _mask(layout, b, t, ["left_arm.joints", "left_arm.gripper"])
    actions = torch.randn(b, t, d) * mask
    tokens, recon = model.encode_decode(actions, mask)
    assert tokens.shape == (b, 4) and tokens.dtype == torch.long
    assert int(tokens.min()) >= 0 and int(tokens.max()) < 16
    assert recon.shape == (b, t, d)
    assert model.num_tokens == 4 and model.vocab_size == 16
    assert model.bits_per_chunk == pytest.approx(4 * math.log2(16))


def test_shorter_horizon_is_accepted(model, layout):
    mask = _mask(layout, 2, 5, ["right_arm.joints"])
    tokens, recon = model.encode_decode(torch.randn(2, 5, layout.total_dim) * mask, mask)
    assert tokens.shape == (2, 4) and recon.shape == (2, 5, layout.total_dim)


def test_padding_values_do_not_influence_the_model(model, layout):
    b, t, d = 2, 10, layout.total_dim
    mask = _mask(layout, b, t, ["left_arm.joints", "left_hand.joints"], time_cut=7)
    actions = torch.randn(b, t, d) * mask
    polluted = actions.clone()
    polluted[~mask] = 1e6
    polluted[0, 9, 0] = float("nan")                       # padded position, must never reach the graph
    tokens_a, recon_a = model.encode_decode(actions, mask)
    tokens_b, recon_b = model.encode_decode(polluted, mask)
    assert torch.equal(tokens_a, tokens_b)
    torch.testing.assert_close(recon_a, recon_b)


def test_decoder_outputs_exact_zeros_in_padding(model, layout):
    mask = _mask(layout, 2, 10, ["torso.joints", "head.joints"], time_cut=6)
    _, recon = model.encode_decode(torch.randn(2, 10, layout.total_dim) * mask, mask)
    assert torch.count_nonzero(recon[~mask]) == 0
    assert recon[mask].abs().sum() > 0


def test_group_queries_only_see_their_own_group(model, layout):
    b, t = 2, 10
    mask = _mask(layout, b, t, ["left_hand.joints", "right_arm.joints"])
    actions = torch.randn(b, t, layout.total_dim) * mask
    changed = actions.clone()
    s, e = layout.intervals()["left_hand.joints"]
    changed[:, :, s:e] += 5.0
    g_hand = layout.names.index("left_hand.joints")
    g_arm = layout.names.index("right_arm.joints")
    step = model.encode_step_tokens(actions, mask, mix=False)             # [B, T, K, d], groups still isolated
    step2 = model.encode_step_tokens(changed, mask, mix=False)
    assert not torch.allclose(step[:, :, g_hand], step2[:, :, g_hand])    # its own group reacts
    torch.testing.assert_close(step[:, :, g_arm], step2[:, :, g_arm])     # a different group does not
    assert not torch.allclose(step[:, :, layout.num_groups:], step2[:, :, layout.num_groups:])  # free queries do
    mixed = model.encode_step_tokens(actions, mask)                       # after the within-step self-attention
    mixed2 = model.encode_step_tokens(changed, mask)
    assert not torch.allclose(mixed[:, :, g_arm], mixed2[:, :, g_arm])    # information crosses groups there


def test_loss_backward_reaches_every_parameter(tiny_model_config, layout):
    torch.manual_seed(0)
    model = HierActionTokenizer(tiny_model_config).train()
    mask = _mask(layout, 4, 10, ["left_arm.joints", "left_arm.gripper", "base.velocity"], time_cut=8)
    actions = torch.randn(4, 10, layout.total_dim) * mask
    out = model(actions, mask)
    assert set(out) >= {"loss", "recon_mse", "aux_loss", "indices", "recon"}
    assert torch.isfinite(out["loss"]) and float(out["recon_mse"]) > 0
    out["loss"].backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
    assert missing == []


def test_group_weights_change_the_loss(tiny_model_config, layout):
    cfg = HierTokenizerConfig.from_dict({**tiny_model_config.to_dict(), "group_weights": {"left_arm.gripper": 10.0}})
    torch.manual_seed(0)
    weighted = HierActionTokenizer(cfg).eval()
    torch.manual_seed(0)
    plain = HierActionTokenizer(tiny_model_config).eval()
    mask = _mask(layout, 2, 10, ["left_arm.joints", "left_arm.gripper"])
    actions = torch.randn(2, 10, layout.total_dim) * mask
    assert float(weighted(actions, mask)["recon_mse"]) != float(plain(actions, mask)["recon_mse"])


def test_detokenize_matches_forward_and_survives_save_load(model, layout, tmp_path):
    mask = _mask(layout, 2, 10, ["right_hand.joints", "right_arm.gripper"])
    actions = torch.randn(2, 10, layout.total_dim) * mask
    tokens, recon = model.encode_decode(actions, mask)
    torch.testing.assert_close(model.detokenize(tokens, mask), recon)
    model.save_pretrained(tmp_path / "ckpt")
    again = HierActionTokenizer.from_pretrained(tmp_path / "ckpt").eval()
    tokens2, recon2 = again.encode_decode(actions, mask)
    assert torch.equal(tokens, tokens2)
    torch.testing.assert_close(recon, recon2)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=.:/Users/artemon/projects/action_chunks /Users/artemon/projects/action_chunks/.venv/bin/pytest -q tests/test_model.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'vqvae_latent_actions.models.hier_tokenizer'`

- [ ] **Step 3: Write the implementation**

```python
# file: vqvae_latent_actions/models/blocks.py
"""Pre-LayerNorm attention blocks built on torch SDPA. Boolean masks: True means "may attend"."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")
        self.heads, self.dropout = int(heads), float(dropout)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, 2 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor, context: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        b, lq, dim = x.shape
        lk = context.shape[1]
        head_dim = dim // self.heads
        q = self.to_q(x).view(b, lq, self.heads, head_dim).transpose(1, 2)
        kv = self.to_kv(context).view(b, lk, 2, self.heads, head_dim).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, kv[0], kv[1], attn_mask=attn_mask,
                                             dropout_p=self.dropout if self.training else 0.0)
        return self.proj(out.transpose(1, 2).reshape(b, lq, dim))


class CrossBlock(nn.Module):
    """x attends to context."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm_q, self.norm_kv = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)

    def forward(self, x: Tensor, context: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        return x + self.attn(self.norm_q(x), self.norm_kv(context), attn_mask)


class SelfBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        normed = self.norm(x)
        return x + self.attn(normed, normed, attn_mask)


class FFBlock(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(nn.Linear(dim, mult * dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(mult * dim, dim), nn.Dropout(dropout))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(self.norm(x))


class PerceiverLayer(nn.Module):
    """cross-attention (optional) -> self-attention (optional) -> feed forward."""

    def __init__(self, dim: int, heads: int, *, cross: bool, self_attention: bool = True, mult: int = 4,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.cross = CrossBlock(dim, heads, dropout) if cross else None
        self.self_attn = SelfBlock(dim, heads, dropout) if self_attention else None
        self.ff = FFBlock(dim, mult, dropout)

    def forward(self, x: Tensor, context: Tensor | None = None, attn_mask: Tensor | None = None,
                self_mask: Tensor | None = None) -> Tensor:
        if self.cross is not None:
            if context is None:
                raise ValueError("cross-attention layer needs a context")
            x = self.cross(x, context, attn_mask)
        if self.self_attn is not None:
            x = self.self_attn(x, self_mask)
        return self.ff(x)


__all__ = ["Attention", "CrossBlock", "SelfBlock", "FFBlock", "PerceiverLayer"]
```

```python
# file: vqvae_latent_actions/models/hier_tokenizer.py
"""Hierarchical tokenizer for unified-action-space chunks.

    actions [B, T, D] + mask [B, T, D]
      -> pointwise embedding (value / learned [PAD], per-dim, per-group, per-time)        [B, T, D, d]
      -> per-step queries (one per semantic group + free queries), group-restricted       [B, T, K, d]
      -> self-attention over T*K tokens (chronological dynamics)                          [B, T*K, d]
      -> N Perceiver latents                                                              [B, N, d]
      -> quantizer                                                                        [B, N] tokens
      -> mirrored decoder conditioned on the same mask, per-dim heads, zeros in padding   [B, T, D]

T (horizon), N (num_tokens) and V (quantizer vocabulary) are independent configuration values.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..layout import UnifiedLayout
from .blocks import PerceiverLayer, SelfBlock, FFBlock
from .quantizers import QuantizerOutput, build_quantizer


@dataclass
class HierTokenizerConfig:
    layout: dict[str, Any]
    horizon: int = 10                 # T the model is trained on (sequences up to max_horizon are accepted)
    max_horizon: int = 64
    num_tokens: int = 10              # N tokens per chunk
    quantizer: dict[str, Any] = field(default_factory=lambda: {"type": "fsq", "levels": [8, 8, 8, 4], "commitment": 0.25})
    dim: int = 256
    heads: int = 8
    free_queries: int = 4             # per-step queries that see every dimension, on top of one query per group
    enc_step_layers: int = 2
    enc_time_layers: int = 4
    enc_latent_layers: int = 4
    dec_latent_layers: int = 2
    dec_layers: int = 4
    ff_mult: int = 4
    dropout: float = 0.0
    value_embedding: str = "mlp"      # "mlp" | "linear"
    group_weights: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HierTokenizerConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})


class HierActionTokenizer(nn.Module):
    def __init__(self, config: HierTokenizerConfig) -> None:
        super().__init__()
        self.config = config
        layout = UnifiedLayout.from_dict(config.layout)
        self.layout = layout
        d, g, f = config.dim, layout.num_groups, config.free_queries
        self.num_dims, self.num_groups, self.num_queries = layout.total_dim, g, g + f
        dims, queries = self.num_dims, self.num_queries

        visibility = torch.cat([layout.membership(), torch.ones(f, dims, dtype=torch.bool)], dim=0)
        self.register_buffer("visibility", visibility, persistent=False)                     # [K, D]
        self.register_buffer("group_index", layout.group_index(), persistent=False)          # [D]
        self.register_buffer("dim_weights", layout.dim_weights(config.group_weights), persistent=False)
        self.register_buffer("visible_counts", visibility.sum(dim=1).clamp_min(1).float(), persistent=False)

        # pointwise embedding
        if config.value_embedding == "linear":
            self.value_proj: nn.Module = nn.Linear(1, d)
        elif config.value_embedding == "mlp":
            self.value_proj = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        else:
            raise ValueError(f"unknown value_embedding {config.value_embedding!r}")
        self.dim_emb = nn.Parameter(torch.randn(dims, d) * 0.02)
        self.pad_emb = nn.Parameter(torch.randn(d) * 0.02)
        self.group_emb = nn.Parameter(torch.randn(g, d) * 0.02)
        self.time_emb = nn.Parameter(torch.randn(config.max_horizon, d) * 0.02)

        # encoder
        self.step_queries = nn.Parameter(torch.randn(queries, d) * 0.02)
        # group queries stay isolated inside the cross-attention stack; the queries of a step mix right after it
        self.enc_step = nn.ModuleList([PerceiverLayer(d, config.heads, cross=True, self_attention=False,
                                                      mult=config.ff_mult, dropout=config.dropout)
                                       for _ in range(config.enc_step_layers)])
        self.enc_step_mix = nn.ModuleList([SelfBlock(d, config.heads, config.dropout),
                                           FFBlock(d, config.ff_mult, config.dropout)])
        self.enc_time = nn.ModuleList([PerceiverLayer(d, config.heads, cross=False, mult=config.ff_mult,
                                                      dropout=config.dropout) for _ in range(config.enc_time_layers)])
        self.latent_queries = nn.Parameter(torch.randn(config.num_tokens, d) * 0.02)
        self.enc_latent = nn.ModuleList([PerceiverLayer(d, config.heads, cross=True, mult=config.ff_mult,
                                                        dropout=config.dropout) for _ in range(config.enc_latent_layers)])
        self.enc_norm = nn.LayerNorm(d)

        self.quantizer = build_quantizer(config.quantizer)
        self.to_code = nn.Linear(d, self.quantizer.code_dim)
        self.from_code = nn.Linear(self.quantizer.out_dim, d)

        # decoder
        self.latent_pos = nn.Parameter(torch.randn(config.num_tokens, d) * 0.02)
        self.dec_latent = nn.ModuleList([nn.ModuleList([SelfBlock(d, config.heads, config.dropout),
                                                        FFBlock(d, config.ff_mult, config.dropout)])
                                         for _ in range(config.dec_latent_layers)])
        self.out_queries = nn.Parameter(torch.randn(queries, d) * 0.02)
        self.mask_emb = nn.Parameter(torch.randn(dims, d) * 0.02)
        self.dec_blocks = nn.ModuleList([PerceiverLayer(d, config.heads, cross=True, mult=config.ff_mult,
                                                        dropout=config.dropout) for _ in range(config.dec_layers)])
        self.dec_norm = nn.LayerNorm(d)
        self.head_group_w = nn.Parameter(torch.randn(dims, d) * (d ** -0.5))
        self.head_free_w = nn.Parameter(torch.randn(dims, d) * (d ** -0.5))
        self.head_bias = nn.Parameter(torch.zeros(dims))
        self.pool_q = nn.Parameter(torch.randn(dims, d) * 0.02)

    # ------------------------------------------------------------------ properties
    @property
    def num_tokens(self) -> int:
        return int(self.config.num_tokens)

    @property
    def vocab_size(self) -> int:
        return int(self.quantizer.vocab_size)

    @property
    def bits_per_chunk(self) -> float:
        return self.num_tokens * self.quantizer.bits_per_token

    # ------------------------------------------------------------------ encoder
    def _check(self, actions: Tensor, mask: Tensor) -> tuple[int, int]:
        if actions.shape != mask.shape:
            raise ValueError(f"actions {tuple(actions.shape)} and mask {tuple(mask.shape)} must have the same shape")
        b, t, d = actions.shape
        if d != self.num_dims:
            raise ValueError(f"expected {self.num_dims} action dims, got {d}")
        if t > self.config.max_horizon:
            raise ValueError(f"horizon {t} exceeds max_horizon {self.config.max_horizon}")
        return b, t

    def _pointwise(self, actions: Tensor, mask: Tensor) -> Tensor:
        """[B, T, D, d] embeddings; padded positions carry the learned [PAD] vector and never the value."""
        values = self.value_proj(actions.masked_fill(~mask, 0.0).unsqueeze(-1))
        embedded = torch.where(mask.unsqueeze(-1), values + self.dim_emb, self.pad_emb)
        embedded = embedded + self.group_emb[self.group_index]
        return embedded + self.time_emb[: actions.shape[1]].unsqueeze(1)

    def encode_step_tokens(self, actions: Tensor, mask: Tensor, mix: bool = True) -> Tensor:
        """[B, T, K, d]: one token per semantic group (restricted to that group's dimensions) plus free tokens.

        With `mix=False` a group token is still a function of its own group only; the within-step self-attention
        that follows is what lets the queries of one timestep exchange information.
        """
        b, t = self._check(actions, mask)
        context = self._pointwise(actions, mask).reshape(b * t, self.num_dims, self.config.dim)
        x = self.step_queries.unsqueeze(0).expand(b * t, -1, -1)
        for layer in self.enc_step:
            x = layer(x, context, attn_mask=self.visibility)
        if mix:
            self_block, ff = self.enc_step_mix
            x = ff(self_block(x))
        return x.reshape(b, t, self.num_queries, self.config.dim)

    def encode_continuous(self, actions: Tensor, mask: Tensor) -> Tensor:
        """[B, N, code_dim]: continuous latents before quantization."""
        b, t = self._check(actions, mask)
        x = self.encode_step_tokens(actions, mask) + self.time_emb[:t].unsqueeze(1)
        x = x.reshape(b, t * self.num_queries, self.config.dim)
        for layer in self.enc_time:
            x = layer(x)
        latents = self.latent_queries.unsqueeze(0).expand(b, -1, -1)
        for layer in self.enc_latent:
            latents = layer(latents, x)
        return self.to_code(self.enc_norm(latents))

    def quantize(self, latents: Tensor) -> QuantizerOutput:
        return self.quantizer(latents)

    # ------------------------------------------------------------------ decoder
    def _presence(self, mask: Tensor) -> Tensor:
        """[B, T, K, d]: what each query is responsible for, given the mask (its group's present dims)."""
        weights = self.visibility.float().unsqueeze(-1) * self.mask_emb.unsqueeze(0)        # [K, D, d]
        presence = torch.einsum("btj,kjd->btkd", mask.float(), weights)
        return presence / self.visible_counts.view(1, 1, -1, 1)

    def decode_latents(self, codes: Tensor, mask: Tensor) -> Tensor:
        b, t, _ = mask.shape
        if t > self.config.max_horizon:
            raise ValueError(f"horizon {t} exceeds max_horizon {self.config.max_horizon}")
        latents = self.from_code(codes) + self.latent_pos
        for self_block, ff in self.dec_latent:
            latents = ff(self_block(latents))
        queries = self.out_queries.view(1, 1, -1, self.config.dim) + self.time_emb[:t].view(1, t, 1, -1)
        queries = queries + self._presence(mask)
        x = queries.reshape(b, t * self.num_queries, self.config.dim)
        for layer in self.dec_blocks:
            x = layer(x, latents)
        x = self.dec_norm(x).reshape(b, t, self.num_queries, self.config.dim)
        group_tokens, free_tokens = x[:, :, : self.num_groups], x[:, :, self.num_groups:]
        per_group = torch.einsum("btgd,jd->btgj", group_tokens, self.head_group_w)           # [B,T,G,D]
        index = self.group_index.view(1, 1, 1, -1).expand(b, t, 1, self.num_dims)
        from_group = per_group.gather(2, index).squeeze(2)                                   # [B,T,D]
        scores = torch.einsum("btfd,jd->btjf", free_tokens, self.pool_q) * (self.config.dim ** -0.5)
        attention = scores.softmax(dim=-1)                                                   # [B,T,D,F]
        per_free = torch.einsum("btfd,jd->btfj", free_tokens, self.head_free_w)              # [B,T,F,D]
        from_free = torch.einsum("btjf,btfj->btj", attention, per_free)
        out = from_group + from_free + self.head_bias
        return out * mask.to(out.dtype)

    # ------------------------------------------------------------------ full passes
    def forward(self, actions: Tensor, mask: Tensor) -> dict[str, Tensor]:
        latents = self.encode_continuous(actions, mask)
        quantized = self.quantize(latents)
        recon = self.decode_latents(quantized.codes, mask)
        target = actions.masked_fill(~mask, 0.0).to(recon.dtype)
        weights = mask.to(recon.dtype) * self.dim_weights.view(1, 1, -1).to(recon.dtype)
        recon_mse = (((recon - target) ** 2) * weights).sum() / weights.sum().clamp_min(1.0)
        loss = recon_mse + quantized.aux_loss.to(recon_mse.dtype)
        return {"loss": loss, "recon_mse": recon_mse.detach(), "aux_loss": quantized.aux_loss.detach(),
                "indices": quantized.indices, "recon": recon, "latents": latents}

    @torch.no_grad()
    def tokenize(self, actions: Tensor, mask: Tensor) -> Tensor:
        return self.quantize(self.encode_continuous(actions, mask)).indices

    @torch.no_grad()
    def detokenize(self, tokens: Tensor, mask: Tensor) -> Tensor:
        return self.decode_latents(self.quantizer.indices_to_codes(tokens), mask)

    @torch.no_grad()
    def encode_decode(self, actions: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        quantized = self.quantize(self.encode_continuous(actions, mask))
        return quantized.indices, self.decode_latents(quantized.codes, mask)

    # ------------------------------------------------------------------ persistence
    def save_pretrained(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(json.dumps(self.config.to_dict(), indent=1))
        torch.save(self.state_dict(), directory / "model.pt")
        return directory

    @classmethod
    def from_pretrained(cls, directory: str | Path, map_location: Any = "cpu") -> "HierActionTokenizer":
        directory = Path(directory)
        config = HierTokenizerConfig.from_dict(json.loads((directory / "config.json").read_text()))
        model = cls(config)
        model.load_state_dict(torch.load(directory / "model.pt", map_location=map_location))
        return model


__all__ = ["HierActionTokenizer", "HierTokenizerConfig"]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=.:/Users/artemon/projects/action_chunks /Users/artemon/projects/action_chunks/.venv/bin/pytest -q tests/test_model.py tests/test_quantizers.py tests/test_layout.py`
Expected: all passed (8 model tests)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "model: hierarchical action tokenizer with group-restricted per-step queries and mask-aware decoder"
```

---

### Task 4: Data layer over action_chunks

**Files:**
- Create: `vqvae_latent_actions/data/__init__.py`, `vqvae_latent_actions/data/chunks.py`, `tests/test_data.py`

**Interfaces:**
- Consumes: `UnifiedLayout` (Task 1); external package `action_chunks` (`Manifest`, `WeightedChunkDataset`, `SequentialChunkDataset`, `collate_chunks`).
- Produces: `batch_to_inputs(batch) -> (actions[B,T,D] float32, mask[B,T,D] bool, embodiment_index[B] long)`;
  `train_loader(manifest, cache_dir, ...) -> DataLoader`; `sequential_loader(...) -> DataLoader`;
  `EvalSet` dataclass with `.batches(size)`; `build_eval_set/save_eval_set/load_eval_set`;
  `layout_from_manifest(path) -> UnifiedLayout`; constant `HOLDOUT_EVERY = 50`.

- [ ] **Step 1: Write the failing test**

```python
# file: tests/test_data.py
import numpy as np
import torch

from vqvae_latent_actions.data.chunks import (batch_to_inputs, build_eval_set, layout_from_manifest, load_eval_set,
                                              save_eval_set, train_loader)


def test_batch_to_inputs_mask_semantics(tiny_manifest, layout):
    loader = train_loader(tiny_manifest, None, batch_size=8, num_workers=0, seed=1, epoch_length=32)
    batch = next(iter(loader))
    actions, mask, emb = batch_to_inputs(batch)
    assert actions.shape == (8, 10, layout.total_dim) and actions.dtype == torch.float32
    assert mask.shape == actions.shape and mask.dtype == torch.bool
    assert emb.shape == (8,) and set(emb.tolist()) <= {0, 1}
    per_step = mask.sum(dim=2)
    assert set(per_step[per_step > 0].tolist()) == {2}          # two real joint dims inside the 119-dim layout
    assert torch.count_nonzero(actions[~mask]) == 0              # padded entries are zeros


def test_train_loader_is_deterministic_for_a_seed(tiny_manifest):
    def first_batch(seed):
        loader = train_loader(tiny_manifest, None, batch_size=4, num_workers=0, seed=seed, epoch_length=16)
        return batch_to_inputs(next(iter(loader)))[0]
    torch.testing.assert_close(first_batch(3), first_batch(3))
    assert not torch.allclose(first_batch(3), first_batch(4))


def test_layout_from_manifest_matches_the_yaml(tiny_manifest, layout):
    assert layout_from_manifest(tiny_manifest).intervals() == layout.intervals()


def test_eval_set_roundtrip_and_cap(tiny_eval_set, tmp_path, layout):
    es = tiny_eval_set
    assert len(es) > 0 and es.actions.shape[1:] == (10, layout.total_dim)
    counts = np.bincount(es.embodiment_index, minlength=2)
    assert counts.tolist() == [6, 6]                             # per_embodiment cap honoured
    assert es.valid.shape == es.actions.shape and es.time_valid.shape == es.actions.shape[:2]
    save_eval_set(es, tmp_path / "e.npz")
    again = load_eval_set(tmp_path / "e.npz")
    np.testing.assert_array_equal(again.actions, es.actions)
    np.testing.assert_array_equal(again.valid, es.valid)
    assert again.embodiment_ids == es.embodiment_ids and again.manifest_sha == es.manifest_sha
    total = sum(len(a) for a, _, _, _ in again.batches(5))
    assert total == len(again)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=.:/Users/artemon/projects/action_chunks /Users/artemon/projects/action_chunks/.venv/bin/pytest -q tests/test_data.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'vqvae_latent_actions.data'`

- [ ] **Step 3: Write the implementation**

```python
# file: vqvae_latent_actions/data/__init__.py
"""Data access for the hierarchical tokenizer (thin layer over the action_chunks package)."""
from .chunks import (EvalSet, HOLDOUT_EVERY, batch_to_inputs, build_eval_set, layout_from_manifest, load_eval_set,
                     load_manifest, save_eval_set, sequential_loader, train_loader)

__all__ = ["EvalSet", "HOLDOUT_EVERY", "batch_to_inputs", "build_eval_set", "layout_from_manifest", "load_eval_set",
           "load_manifest", "save_eval_set", "sequential_loader", "train_loader"]
```

```python
# file: vqvae_latent_actions/data/chunks.py
"""Chunks of the R0_v2.1 mixture as the VLA sees them: actions [T, 119] plus a {0,1} mask of real entries.

Training samples with the VLA mixture weights (WeightedChunkDataset); the shared eval set is a deterministic,
evenly spread pass over held-out episodes, stored as npz so every tokenizer is scored on the very same chunks.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from action_chunks.datasets import SequentialChunkDataset, WeightedChunkDataset, collate_chunks
from action_chunks.manifest import Manifest

from ..layout import UnifiedLayout

HOLDOUT_EVERY = 50   # embodiments without a validation list hold out every 50th episode (2%)


def load_manifest(manifest: str | Path | Manifest) -> Manifest:
    return manifest if isinstance(manifest, Manifest) else Manifest.load(manifest)


def layout_from_manifest(manifest: str | Path | Manifest) -> UnifiedLayout:
    return UnifiedLayout.from_manifest(load_manifest(manifest))


def batch_to_inputs(batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(actions, mask, embodiment_index). Mask is True only where the value is a real action."""
    actions = batch["fast_action"].float()
    time_valid = ~batch["fast_action_is_pad"].bool()
    mask = ~batch["fast_action_pad_mask"].bool() & time_valid.unsqueeze(-1)
    actions = actions.masked_fill(~mask, 0.0)
    return actions, mask, batch["embodiment_index"].long()


def train_loader(manifest: str | Path, cache_dir: str | Path | None, *, batch_size: int, num_workers: int,
                 seed: int = 0, samples_per_episode: int = 8, epoch_length: int | None = None, split: str = "train",
                 holdout_every: int | None = HOLDOUT_EVERY, pin_memory: bool = False) -> DataLoader:
    dataset = WeightedChunkDataset(load_manifest(manifest), split=split, cache_dir=cache_dir, seed=seed,
                                   samples_per_episode=samples_per_episode, epoch_length=epoch_length,
                                   holdout_every=holdout_every)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_chunks,
                      drop_last=True, pin_memory=pin_memory, persistent_workers=num_workers > 0)


def sequential_loader(manifest: str | Path, cache_dir: str | Path | None, *, split: str = "val", stride: int = 1,
                      max_chunks_per_embodiment: int | None = None, batch_size: int = 256, num_workers: int = 8,
                      embodiments: list[str] | None = None, holdout_every: int | None = HOLDOUT_EVERY) -> DataLoader:
    dataset = SequentialChunkDataset(load_manifest(manifest), stride=stride, split=split, cache_dir=cache_dir,
                                     embodiments=embodiments, max_chunks_per_embodiment=max_chunks_per_embodiment,
                                     holdout_every=holdout_every)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_chunks)


@dataclass
class EvalSet:
    actions: np.ndarray            # [N, T, D] float32
    valid: np.ndarray              # [N, T, D] bool
    time_valid: np.ndarray         # [N, T] bool
    embodiment_index: np.ndarray   # [N] int64
    embodiment_ids: list[str]
    manifest_sha: str

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    def batches(self, batch_size: int):
        for start in range(0, len(self), batch_size):
            sl = slice(start, start + batch_size)
            yield self.actions[sl], self.valid[sl], self.time_valid[sl], self.embodiment_index[sl]


def build_eval_set(manifest: str | Path, cache_dir: str | Path | None, out_path: str | Path, *,
                   per_embodiment: int = 500, split: str = "val", num_workers: int = 8, batch_size: int = 256,
                   seed: int = 0, holdout_every: int | None = HOLDOUT_EVERY) -> EvalSet:
    """At most `per_embodiment` evenly spread chunks from the held-out episodes of every embodiment."""
    manifest = load_manifest(manifest)
    rng = np.random.default_rng(seed)
    parts: dict[str, list[np.ndarray]] = {k: [] for k in ("actions", "valid", "time_valid", "emb")}
    for spec in manifest.embodiments:
        loader = sequential_loader(manifest, cache_dir, split=split, stride=1, max_chunks_per_embodiment=per_embodiment,
                                   batch_size=batch_size, num_workers=num_workers, embodiments=[spec.embodiment_id],
                                   holdout_every=holdout_every)
        got: dict[str, list[np.ndarray]] = {k: [] for k in parts}
        for batch in loader:
            actions, mask, emb = batch_to_inputs(batch)
            got["actions"].append(actions.numpy())
            got["valid"].append(mask.numpy())
            got["time_valid"].append((~batch["fast_action_is_pad"].bool()).numpy())
            got["emb"].append(emb.numpy())
        if not got["actions"]:
            print(f"eval set: {spec.embodiment_id:<48} {0:>6} chunks (no {split} episodes)", flush=True)
            continue
        merged = {k: np.concatenate(v) for k, v in got.items()}
        found = len(merged["actions"])
        if found > per_embodiment:
            keep = np.sort(rng.choice(found, size=per_embodiment, replace=False))
            merged = {k: v[keep] for k, v in merged.items()}
        for key in parts:
            parts[key].append(merged[key])
        print(f"eval set: {spec.embodiment_id:<48} {len(merged['actions']):>6} chunks (pass yielded {found})", flush=True)
    eval_set = EvalSet(actions=np.concatenate(parts["actions"]), valid=np.concatenate(parts["valid"]),
                       time_valid=np.concatenate(parts["time_valid"]), embodiment_index=np.concatenate(parts["emb"]),
                       embodiment_ids=[e.embodiment_id for e in manifest.embodiments], manifest_sha=manifest.sha256)
    save_eval_set(eval_set, out_path)
    return eval_set


def save_eval_set(eval_set: EvalSet, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, actions=eval_set.actions, valid=eval_set.valid, time_valid=eval_set.time_valid,
                        embodiment_index=eval_set.embodiment_index,
                        embodiment_ids=np.array(json.dumps(eval_set.embodiment_ids)),
                        manifest_sha=np.array(eval_set.manifest_sha))


def load_eval_set(path: str | Path) -> EvalSet:
    z = np.load(path)
    return EvalSet(actions=z["actions"].astype(np.float32), valid=z["valid"].astype(bool),
                   time_valid=z["time_valid"].astype(bool), embodiment_index=z["embodiment_index"].astype(np.int64),
                   embodiment_ids=json.loads(str(z["embodiment_ids"])), manifest_sha=str(z["manifest_sha"]))


__all__ = ["EvalSet", "HOLDOUT_EVERY", "batch_to_inputs", "build_eval_set", "layout_from_manifest", "load_eval_set",
           "load_manifest", "save_eval_set", "sequential_loader", "train_loader"]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONPATH=.:/Users/artemon/projects/action_chunks /Users/artemon/projects/action_chunks/.venv/bin/pytest -q tests/test_data.py`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "data: weighted training chunks and the shared eval set over action_chunks"
```

---

### Task 5: Metrics and evaluation

**Files:**
- Create: `vqvae_latent_actions/training/metrics.py`, `vqvae_latent_actions/training/evaluate.py`, `tests/test_metrics.py`, `tests/test_evaluate.py`
- Delete: `vqvae_latent_actions/training/eval.py`, `vqvae_latent_actions/utils/logging.py`, `vqvae_latent_actions/utils/metrics.py`, `vqvae_latent_actions/utils/__init__.py`

**Interfaces:**
- Consumes: `EvalSet` (Task 4), model API (Task 3), `code_usage` (Task 2).
- Produces: `masked_errors(actions, recon, valid) -> {mse, l1, max_abs}`; `MetricAccumulator(embodiment_ids)` with
  `.add(actions, recon, valid, embodiment_index, token_lengths, failed=None)` and `.summary()`;
  `write_report(path, arm, summary, extra=None)`; `evaluate_tokenizer(model, eval_set, batch_size, device) -> summary`
  (adds `summary["usage"]`); `padding_invariance_mismatch(model, eval_set, ...) -> float`; `model_report_extra(model) -> dict`.

- [ ] **Step 1: Write the failing tests**

```python
# file: tests/test_metrics.py
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
```

```python
# file: tests/test_evaluate.py
import torch

from vqvae_latent_actions.models.hier_tokenizer import HierActionTokenizer
from vqvae_latent_actions.training.evaluate import (evaluate_tokenizer, model_report_extra,
                                                    padding_invariance_mismatch)


def test_evaluate_tokenizer_reports_errors_and_usage(tiny_model_config, tiny_eval_set, device):
    torch.manual_seed(0)
    model = HierActionTokenizer(tiny_model_config).eval()
    summary = evaluate_tokenizer(model, tiny_eval_set, batch_size=4, device=device)
    total = summary["total"]
    assert total["n"] == len(tiny_eval_set)
    assert total["tokens_mean"] == model.num_tokens and total["decode_failures"] == 0
    assert total["rmse"] > 0 and set(summary["per_embodiment"]) == {"toy_joint_0", "toy_joint_1"}
    usage = summary["usage"]
    assert usage["vocab_size"] == model.vocab_size and usage["tokens_per_chunk"] == model.num_tokens
    assert usage["bits_per_chunk"] == model.bits_per_chunk
    assert 0 < usage["usage_percent"] <= 100 and usage["perplexity"] >= 1


def test_padding_invariance_is_exact(tiny_model_config, tiny_eval_set, device):
    torch.manual_seed(0)
    model = HierActionTokenizer(tiny_model_config).eval()
    assert padding_invariance_mismatch(model, tiny_eval_set, device=device) == 0.0


def test_model_report_extra(tiny_model_config):
    model = HierActionTokenizer(tiny_model_config)
    extra = model_report_extra(model)
    assert extra["num_tokens"] == 4 and extra["vocab_size"] == 16 and extra["bits_per_chunk"] == 16.0
    assert extra["quantizer"]["type"] == "fsq" and extra["parameters"] > 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=.:/Users/artemon/projects/action_chunks /Users/artemon/projects/action_chunks/.venv/bin/pytest -q tests/test_metrics.py tests/test_evaluate.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'vqvae_latent_actions.training.metrics'`

- [ ] **Step 3: Write the implementation**

```python
# file: vqvae_latent_actions/training/metrics.py
"""Reconstruction and token-length metrics over the valid (unpadded) entries, per embodiment.

Kept byte-for-byte compatible with `tokenizer_arms.metrics` so results drop straight into the arm comparison.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def masked_errors(actions: np.ndarray, recon: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    """Per-chunk errors over the valid [B,T,D] entries: mse, l1, max_abs (arrays [B])."""
    mask = np.asarray(valid, dtype=np.float64)
    diff = (np.asarray(recon, dtype=np.float64) - np.asarray(actions, dtype=np.float64))
    n = np.maximum(mask.sum(axis=(1, 2)), 1.0)
    mse = ((diff ** 2) * mask).sum(axis=(1, 2)) / n
    l1 = (np.abs(diff) * mask).sum(axis=(1, 2)) / n
    max_abs = (np.abs(diff) * mask).max(axis=(1, 2))
    return {"mse": mse, "l1": l1, "max_abs": max_abs}


class MetricAccumulator:
    def __init__(self, embodiment_ids: list[str]):
        self.embodiment_ids = list(embodiment_ids)
        self.mse, self.l1, self.max_abs, self.tokens, self.emb, self.failed = [], [], [], [], [], []

    def add(self, actions, recon, valid, embodiment_index, token_lengths, failed=None) -> None:
        e = masked_errors(actions, recon, valid)
        self.mse.append(e["mse"]); self.l1.append(e["l1"]); self.max_abs.append(e["max_abs"])
        self.tokens.append(np.asarray(token_lengths, dtype=np.int64))
        self.emb.append(np.asarray(embodiment_index, dtype=np.int64))
        self.failed.append(np.zeros(len(actions), dtype=bool) if failed is None else np.asarray(failed, dtype=bool))

    @staticmethod
    def _stats(mse, l1, max_abs, tokens, failed) -> dict:
        return {"n": int(len(mse)), "mse": float(mse.mean()), "rmse": float(np.sqrt(mse.mean())), "l1": float(l1.mean()),
                "max_abs_mean": float(max_abs.mean()), "max_abs_max": float(max_abs.max()),
                "tokens_mean": float(tokens.mean()), "tokens_std": float(tokens.std()), "tokens_p50": float(np.median(tokens)),
                "tokens_p95": float(np.percentile(tokens, 95)), "tokens_max": int(tokens.max()),
                "decode_failures": int(failed.sum())}

    def summary(self) -> dict:
        mse, l1, max_abs = np.concatenate(self.mse), np.concatenate(self.l1), np.concatenate(self.max_abs)
        tokens, emb, failed = np.concatenate(self.tokens), np.concatenate(self.emb), np.concatenate(self.failed)
        per = {}
        for i, eid in enumerate(self.embodiment_ids):
            sel = emb == i
            if sel.any():
                per[eid] = self._stats(mse[sel], l1[sel], max_abs[sel], tokens[sel], failed[sel])
        return {"total": self._stats(mse, l1, max_abs, tokens, failed), "per_embodiment": per}


def write_report(path: str | Path, arm: str, summary: dict, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"arm": arm, **(extra or {}), **summary}
    path.write_text(json.dumps(payload, indent=1))
    t = summary["total"]
    lines = [f"# {arm}", "", f"chunks={t['n']} tokens/chunk mean={t['tokens_mean']:.2f} p95={t['tokens_p95']:.0f} max={t['tokens_max']} "
             f"| rmse={t['rmse']:.5f} l1={t['l1']:.5f} max_abs(mean)={t['max_abs_mean']:.4f} decode_failures={t['decode_failures']}", "",
             "| embodiment | n | tokens | rmse | l1 | max_abs |", "|---|---|---|---|---|---|"]
    for eid, s in summary["per_embodiment"].items():
        lines.append(f"| {eid} | {s['n']} | {s['tokens_mean']:.1f} | {s['rmse']:.5f} | {s['l1']:.5f} | {s['max_abs_mean']:.4f} |")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")


__all__ = ["masked_errors", "MetricAccumulator", "write_report"]
```

```python
# file: vqvae_latent_actions/training/evaluate.py
"""Scoring a tokenizer on the shared eval set: reconstruction error, codebook usage, padding invariance."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..data.chunks import EvalSet
from ..models.quantizers import code_usage
from .metrics import MetricAccumulator


@torch.no_grad()
def evaluate_tokenizer(model, eval_set: EvalSet, *, batch_size: int = 1024, device: Any = None) -> dict:
    """Reconstruction metrics per embodiment plus codebook usage; token length is constant (N)."""
    device = torch.device(device) if device is not None else next(model.parameters()).device
    was_training = model.training
    model.eval()
    accumulator = MetricAccumulator(eval_set.embodiment_ids)
    counts = torch.zeros(model.vocab_size, dtype=torch.long)
    for actions, valid, _time_valid, emb in eval_set.batches(batch_size):
        x = torch.as_tensor(actions, device=device)
        m = torch.as_tensor(valid, device=device)
        tokens, recon = model.encode_decode(x, m)
        accumulator.add(actions, recon.float().cpu().numpy(), valid, emb, [model.num_tokens] * len(actions))
        counts += torch.bincount(tokens.reshape(-1).cpu(), minlength=model.vocab_size)
    summary = accumulator.summary()
    used = int((counts > 0).sum())
    probs = counts.float() / counts.sum().clamp_min(1)
    nonzero = probs[probs > 0]
    summary["usage"] = {"vocab_size": model.vocab_size, "tokens_per_chunk": model.num_tokens,
                        "bits_per_chunk": model.bits_per_chunk, "codes_used": used,
                        "usage_percent": 100.0 * used / model.vocab_size,
                        "perplexity": float(torch.exp(-(nonzero * nonzero.log()).sum()))}
    if was_training:
        model.train()
    return summary


@torch.no_grad()
def padding_invariance_mismatch(model, eval_set: EvalSet, *, num: int = 256, batch_size: int = 256,
                                device: Any = None, magnitude: float = 1e3, seed: int = 0) -> float:
    """Fraction of chunks whose tokens change when padded entries are filled with garbage (must be 0)."""
    device = torch.device(device) if device is not None else next(model.parameters()).device
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(seed)
    total = mismatched = 0
    for start in range(0, min(num, len(eval_set)), batch_size):
        sl = slice(start, min(start + batch_size, num, len(eval_set)))
        actions, valid = eval_set.actions[sl], eval_set.valid[sl]
        polluted = actions.copy()
        noise = rng.normal(scale=magnitude, size=actions.shape).astype(np.float32)
        polluted[~valid] = noise[~valid]
        m = torch.as_tensor(valid, device=device)
        a = model.tokenize(torch.as_tensor(actions, device=device), m)
        b = model.tokenize(torch.as_tensor(polluted, device=device), m)
        mismatched += int((a != b).any(dim=-1).sum())
        total += a.shape[0]
    if was_training:
        model.train()
    return float(mismatched) / max(total, 1)


def model_report_extra(model) -> dict:
    """Static facts about the tokenizer that belong in every report."""
    return {"num_tokens": model.num_tokens, "vocab_size": model.vocab_size, "bits_per_chunk": model.bits_per_chunk,
            "horizon": model.config.horizon, "quantizer": dict(model.config.quantizer),
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "dim": model.config.dim, "free_queries": model.config.free_queries}


__all__ = ["evaluate_tokenizer", "padding_invariance_mismatch", "model_report_extra"]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=.:/Users/artemon/projects/action_chunks:/Users/artemon/projects/tokenizer_arms /Users/artemon/projects/action_chunks/.venv/bin/pytest -q tests/test_metrics.py tests/test_evaluate.py`
Expected: 6 passed (the tokenizer_arms comparison runs because the repo is on PYTHONPATH)

- [ ] **Step 5: Delete the superseded training/utils modules and commit**

```bash
git rm -q vqvae_latent_actions/training/eval.py vqvae_latent_actions/utils/logging.py vqvae_latent_actions/utils/metrics.py
git add -A
git commit -m "metrics: masked per-embodiment errors, codebook usage and padding-invariance check"
```

---

### Task 6: Comet logging and the training loop

**Files:**
- Create: `vqvae_latent_actions/training/comet.py`, `tests/test_comet.py`, `tests/test_loop.py`
- Modify (rewrite): `vqvae_latent_actions/training/loop.py`, `vqvae_latent_actions/training/__init__.py`

**Interfaces:**
- Consumes: everything from Tasks 1-5.
- Produces: `RunLogger(mode, project, workspace, experiment_name, tags, offline_directory, jsonl_path, config, enabled)`
  with `.mode`, `.log_params(dict)`, `.log_metrics(dict, step)`, `.log_other(key, value)`, `.end()`;
  `TrainConfig` dataclass; `train(cfg: TrainConfig) -> dict` (final eval summary); `build_model(cfg, layout)`; `lr_lambda(cfg)`.

- [ ] **Step 1: Write the failing tests**

```python
# file: tests/test_comet.py
import json

from vqvae_latent_actions.training.comet import RunLogger


def test_disabled_logger_still_writes_jsonl(tmp_path):
    logger = RunLogger(mode="disabled", jsonl_path=tmp_path / "log.jsonl", config={"a": 1})
    assert logger.mode == "disabled"
    logger.log_params({"lr": 0.1})
    logger.log_metrics({"loss": 1.5}, step=3)
    logger.log_metrics({"loss": 1.0}, step=4)
    logger.end()
    rows = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [3, 4] and rows[-1]["loss"] == 1.0


def test_offline_mode_creates_an_archive(tmp_path):
    import pytest
    pytest.importorskip("comet_ml")
    logger = RunLogger(mode="offline", project="unit-test", offline_directory=tmp_path / "offline",
                       jsonl_path=tmp_path / "log.jsonl", experiment_name="unit")
    assert logger.mode == "offline"
    logger.log_params({"n": 4})
    logger.log_metrics({"loss": 0.5}, step=1)
    logger.end()
    assert list((tmp_path / "offline").glob("*.zip"))
```

```python
# file: tests/test_loop.py
import json

import torch

from vqvae_latent_actions.training.loop import TrainConfig, lr_lambda, train


def _config(tmp_path, tiny_manifest, eval_path, steps: int) -> TrainConfig:
    return TrainConfig(
        out_dir=str(tmp_path / "run"), manifest=str(tiny_manifest), eval_set=str(eval_path), cache_dir=str(tmp_path / "cache"),
        model={"horizon": 10, "max_horizon": 16, "num_tokens": 4, "dim": 32, "heads": 4, "free_queries": 2,
               "enc_step_layers": 1, "enc_time_layers": 1, "enc_latent_layers": 1, "dec_latent_layers": 1, "dec_layers": 1,
               "quantizer": {"type": "fsq", "levels": [4, 4]}},
        steps=steps, batch_size=4, lr=1e-3, warmup_steps=2, num_workers=0, log_every=1, eval_every=2, ckpt_every=2,
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
    cfg = _config(tmp_path, tiny_manifest, eval_path, steps=4)
    train(cfg)
    rows = [json.loads(l) for l in (tmp_path / "run" / "train_log.jsonl").read_text().splitlines()]
    schedule = lr_lambda(cfg)
    for row in rows:
        assert abs(row["lr"] - cfg.lr * schedule(row["step"])) < 1e-9
    assert rows[0]["lr"] < rows[1]["lr"] and rows[-1]["lr"] < rows[1]["lr"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=.:/Users/artemon/projects/action_chunks /Users/artemon/projects/action_chunks/.venv/bin/pytest -q tests/test_comet.py tests/test_loop.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'vqvae_latent_actions.training.comet'`

- [ ] **Step 3: Write the implementation**

```python
# file: vqvae_latent_actions/training/comet.py
"""Comet ML tracking: online when the API is reachable, an offline archive otherwise, disabled in tests.

The API key never lives in the repo: comet_ml picks it up from COMET_API_KEY or the config file pointed to by
COMET_CONFIG (`~/.comet.config` by default). Every metric is mirrored into a local jsonl regardless of the mode.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

MODES = ("auto", "online", "offline", "disabled")


class RunLogger:
    def __init__(self, *, mode: str = "auto", project: str | None = None, workspace: str | None = None,
                 experiment_name: str | None = None, tags: list[str] | None = None,
                 offline_directory: str | Path | None = None, jsonl_path: str | Path | None = None,
                 config: Mapping[str, Any] | None = None, enabled: bool = True) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        if self.jsonl_path:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.experiment = None
        self._mode = "disabled"
        if not enabled or mode == "disabled":
            return
        try:
            import comet_ml
        except Exception as exc:                                    # pragma: no cover - depends on the environment
            print(f"[comet] comet_ml unavailable ({type(exc).__name__}), logging to jsonl only", flush=True)
            return
        resolved = mode
        if mode == "auto":
            resolved = "online" if self._reachable() else "offline"
        kwargs = dict(project_name=project, auto_metric_logging=False, auto_param_logging=False,
                      auto_output_logging="simple", log_code=False, log_graph=False, log_env_details=True)
        try:
            if resolved == "online":
                self.experiment = comet_ml.Experiment(workspace=workspace, **kwargs)
            else:
                directory = Path(offline_directory or "comet_offline")
                directory.mkdir(parents=True, exist_ok=True)
                self.experiment = comet_ml.OfflineExperiment(offline_directory=str(directory), workspace=workspace, **kwargs)
            self._mode = resolved
        except Exception as exc:                                    # pragma: no cover - network/credentials
            print(f"[comet] could not start a {resolved} experiment ({type(exc).__name__}: {exc}); jsonl only", flush=True)
            return
        if experiment_name:
            self.experiment.set_name(experiment_name)
        if tags:
            self.experiment.add_tags(list(tags))
        if config:
            self.experiment.log_parameters(_flatten(config))
        print(f"[comet] mode={self._mode} project={project} workspace={workspace}", flush=True)

    @staticmethod
    def _reachable(url: str = "https://www.comet.com", timeout: float = 5.0) -> bool:
        try:
            import requests

            requests.head(url, timeout=timeout)
            return True
        except Exception:
            return False

    @property
    def mode(self) -> str:
        return self._mode

    def log_params(self, params: Mapping[str, Any]) -> None:
        if self.experiment is not None:
            self.experiment.log_parameters(_flatten(params))

    def log_metrics(self, metrics: Mapping[str, Any], step: int) -> None:
        if self.experiment is not None:
            self.experiment.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))}, step=step)
        if self.jsonl_path:
            with self.jsonl_path.open("a") as handle:
                handle.write(json.dumps({"step": int(step), **{k: v for k, v in metrics.items()}}) + "\n")

    def log_other(self, key: str, value: Any) -> None:
        if self.experiment is not None:
            self.experiment.log_other(key, value)

    def end(self) -> None:
        if self.experiment is not None:
            self.experiment.end()


def _flatten(payload: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.update(_flatten(value, prefix=f"{name}."))
        elif isinstance(value, (list, tuple)):
            out[name] = json.dumps(list(value))
        else:
            out[name] = value
    return out


__all__ = ["RunLogger"]
```

```python
# file: vqvae_latent_actions/training/loop.py
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
    decay = [p for _, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for _, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                                   {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr, betas=tuple(cfg.betas))
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

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    loader = train_loader(cfg.manifest, cfg.cache_dir, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                          seed=cfg.seed * 1000 + rank, samples_per_episode=cfg.samples_per_episode,
                          epoch_length=cfg.batch_size * (cfg.steps + 1), pin_memory=device.type == "cuda")
    loader.dataset.set_epoch(step)

    logger = RunLogger(enabled=accelerator.is_main_process, jsonl_path=out / "train_log.jsonl",
                       experiment_name=cfg.run_name, config={"train": asdict(cfg)},
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
        summary = evaluate_tokenizer(target, eval_set, batch_size=cfg.eval_batch_size, device=device)
        summary["padding_invariance_mismatch"] = padding_invariance_mismatch(target, eval_set, device=device)
        write_report(out / f"eval_step{current:07d}.json", f"{cfg.run_name}@{current}", summary,
                     {"step": current, **model_report_extra(target)})
        total, usage = summary["total"], summary["usage"]
        logger.log_metrics({"eval/rmse": total["rmse"], "eval/l1": total["l1"], "eval/max_abs": total["max_abs_mean"],
                            "eval/codes_used": usage["codes_used"], "eval/perplexity": usage["perplexity"],
                            "eval/padding_mismatch": summary["padding_invariance_mismatch"]}, step=current)
        accelerator.print(f"eval step {current}: rmse={total['rmse']:.5f} l1={total['l1']:.5f} "
                          f"codes={usage['codes_used']}/{usage['vocab_size']} perplexity={usage['perplexity']:.0f}")
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
        with accelerator.autocast():
            output = model(actions, mask)
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
            record = {"step": step, "loss": loss, "recon_mse": float(output["recon_mse"]),
                      "aux_loss": float(output["aux_loss"]), "lr": scheduler.get_last_lr()[0],
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
```

```python
# file: vqvae_latent_actions/training/__init__.py
"""Training, evaluation and tracking."""
from .comet import RunLogger
from .evaluate import evaluate_tokenizer, model_report_extra, padding_invariance_mismatch
from .loop import TrainConfig, build_model, lr_lambda, train
from .metrics import MetricAccumulator, masked_errors, write_report

__all__ = ["RunLogger", "TrainConfig", "train", "build_model", "lr_lambda", "evaluate_tokenizer",
           "padding_invariance_mismatch", "model_report_extra", "MetricAccumulator", "masked_errors", "write_report"]
```

- [ ] **Step 4: Install comet_ml locally (dev) and run the tests**

```bash
/Users/artemon/projects/action_chunks/.venv/bin/pip install -q comet_ml
PYTHONPATH=.:/Users/artemon/projects/action_chunks /Users/artemon/projects/action_chunks/.venv/bin/pytest -q
```
Expected: every test passes (comet tests included, offline archive written to a temp dir)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "training: Comet-backed step trainer with resume, periodic eval and final export"
```

---

### Task 7: Entry point, configs, launchers, docs

**Files:**
- Create: `train.py` (rewrite), `configs/config.yaml`, `configs/model/hier_{fsq,vq,lfq,gumbel}.yaml`, `configs/data/r0_v2_1.yaml`,
  `launchers/{common.sh,train.sh,smoke.sh,submit.sh,comet_upload.sh}`, `docs/runs.md`, `tests/test_entry.py`
- Modify: `README.md`
- Delete: `train.sh` (old ClearML launcher)

**Interfaces:**
- Consumes: `TrainConfig`, `train` (Task 6).
- Produces: hydra config tree; `train.py` CLI; shell launchers with `$NTOK`, `$LEVELS`, `$RUN_NAME`, `$STEPS`, `$NPROC` overrides.

- [ ] **Step 1: Write the failing test**

```python
# file: tests/test_entry.py
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent


def test_config_tree_composes_and_matches_the_budget():
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=["run_name=unit"])
    payload = OmegaConf.to_container(cfg, resolve=True)
    assert payload["model"]["num_tokens"] == 10
    assert payload["model"]["quantizer"]["levels"] == [8, 8, 8, 4]      # 8*8*8*4 = 2048 codes, 11 bits
    assert payload["train"]["steps"] == 300000 and payload["train"]["mixed_precision"] == "bf16"
    assert payload["comet"]["project"] and payload["comet"]["workspace"] == "dont4rootme"
    assert "manifest" in payload["data"] and "eval_set" in payload["data"]


def test_overrides_reach_the_train_config(tmp_path, tiny_manifest, tiny_eval_set):
    from vqvae_latent_actions.data.chunks import save_eval_set
    from train import build_train_config
    eval_path = tmp_path / "eval.npz"
    save_eval_set(tiny_eval_set, eval_path)
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=[
            "run_name=unit", "model=hier_vq", "model.num_tokens=6", "train.steps=7", "train.batch_size=4",
            f"train.out_dir={tmp_path / 'run'}", f"data.manifest={tiny_manifest}", f"data.eval_set={eval_path}",
            "data.cache_dir=null", "comet.mode=disabled"])
    train_cfg = build_train_config(cfg)
    assert train_cfg.steps == 7 and train_cfg.batch_size == 4 and train_cfg.run_name == "unit"
    assert train_cfg.model["num_tokens"] == 6 and train_cfg.model["quantizer"]["type"] == "vq"
    assert train_cfg.comet["mode"] == "disabled" and train_cfg.cache_dir is None
    assert json.loads(json.dumps(train_cfg.model))                       # plain containers, no omegaconf leftovers
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=.:/Users/artemon/projects/action_chunks /Users/artemon/projects/action_chunks/.venv/bin/pytest -q tests/test_entry.py`
Expected: FAIL (no `configs/config.yaml` with the new tree / no `build_train_config` in `train.py`)

- [ ] **Step 3: Write the implementation**

```python
# file: train.py
"""Hydra entry point.

    python train.py run_name=hier-fsq-n10-v2048                       # defaults: FSQ, N=10, V=2048, 300k steps
    python train.py model=hier_vq model.num_tokens=16 train.steps=100000
"""
from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from vqvae_latent_actions.training.loop import TrainConfig, train


def build_train_config(cfg: DictConfig) -> TrainConfig:
    payload = OmegaConf.to_container(cfg, resolve=True)
    train_section = dict(payload["train"])
    data = payload["data"]
    return TrainConfig(manifest=data["manifest"], eval_set=data["eval_set"], cache_dir=data.get("cache_dir"),
                       model=dict(payload["model"]), comet=dict(payload.get("comet", {})), **train_section)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    train(build_train_config(cfg))


if __name__ == "__main__":
    main()
```

```yaml
# file: configs/config.yaml
defaults:
  - model: hier_fsq
  - data: r0_v2_1
  - _self_

run_name: hier-fsq-n10-v2048

train:
  out_dir: ${oc.env:VQLA_OUT,outputs}/${run_name}
  run_name: ${run_name}
  steps: 300000
  batch_size: 256          # per process
  lr: 3.0e-4
  min_lr_ratio: 0.1
  weight_decay: 0.01
  betas: [0.9, 0.95]
  warmup_steps: 2000
  schedule: cosine         # cosine | constant
  grad_clip: 1.0
  num_workers: 10
  samples_per_episode: 8
  seed: 0
  log_every: 100
  eval_every: 10000
  ckpt_every: 5000
  eval_batch_size: 1024
  mixed_precision: bf16

comet:
  mode: auto               # auto | online | offline | disabled
  workspace: dont4rootme
  project: hier-action-tokenizer
  tags: [hier-tokenizer, unified-119]

hydra:
  run:
    dir: ${train.out_dir}/hydra
  job_logging:
    root:
      level: INFO
```

```yaml
# file: configs/model/hier_fsq.yaml
horizon: 10
max_horizon: 64
num_tokens: 10             # N tokens per chunk
dim: 256
heads: 8
free_queries: 4            # per-step queries on top of one query per semantic group
enc_step_layers: 2
enc_time_layers: 4
enc_latent_layers: 4
dec_latent_layers: 2
dec_layers: 4
ff_mult: 4
dropout: 0.0
value_embedding: mlp
group_weights: null
quantizer:
  type: fsq
  levels: [8, 8, 8, 4]     # V = 2048 codes = 11 bits per token
  commitment: 0.25
```

```yaml
# file: configs/model/hier_vq.yaml
defaults:
  - hier_fsq

quantizer:
  type: vq
  vocab_size: 2048
  code_dim: 64
  commitment: 0.25
  decay: 0.99
```

```yaml
# file: configs/model/hier_lfq.yaml
defaults:
  - hier_fsq

quantizer:
  type: lfq
  vocab_size: 2048
  entropy_loss_weight: 0.1
  diversity_gamma: 1.0
```

```yaml
# file: configs/model/hier_gumbel.yaml
defaults:
  - hier_fsq

quantizer:
  type: gumbel
  vocab_size: 2048
  code_dim: 64
  out_dim: 64
  temperature: 1.0
  entropy_weight: 0.01
```

```yaml
# file: configs/data/r0_v2_1.yaml
# R0_v2.1 mixture through action_chunks. Paths default to the cluster and can be overridden by environment.
manifest: ${oc.env:VQLA_MANIFEST,/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/projects/action_tokenization/action_chunks/manifests/r0_v2.1-6885099}
cache_dir: ${oc.env:VQLA_CACHE,/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/projects/action_tokenization/action_chunks_index_cache}
eval_set: ${oc.env:VQLA_EVALSET,/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/projects/action_tokenization/runs/eval_set/eval_set-05d18b2b-val500.npz}
```

```bash
# file: launchers/common.sh
# Shared environment for runs on the Sber cluster (sourced by every launcher).
BASE=/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/projects/action_tokenization
REPO=${REPO:-$BASE/vqvae_lattent_actions}
ACTION_CHUNKS=$BASE/action_chunks
RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
trap 'rc=$?; echo "[launch] exit=$rc at $(date -Is)"; echo "$rc" > "$RUN_DIR/exit_code"; exit $rc' EXIT
echo "[launch] start $(date -Is) host=$(hostname) run_dir=$RUN_DIR"
export MAMBA_EXE=/mnt/virtual_ai0001071-01239_SR006-nfs2/.local/bin/micromamba
export MAMBA_ROOT_PREFIX=/mnt/virtual_ai0001071-01239_SR006-nfs2/micromamba
test -x "$MAMBA_EXE" || { echo "[launch] micromamba missing"; exit 1; }
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate ai_lerobot_qwen3vl_develop || { echo "[launch] env activation failed"; exit 1; }
set -e -o pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO:$ACTION_CHUNKS:$BASE/pylib:$BASE/pylib_comet"
export COMET_CONFIG=/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/.comet.config   # key lives here, never in git
export VQLA_OUT="${VQLA_OUT:-$RUN_DIR/out}"
echo "[launch] python=$(python -c 'import sys;print(sys.executable)') repo=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo n/a) action_chunks=$(git -C "$ACTION_CHUNKS" rev-parse --short HEAD 2>/dev/null || echo n/a)"
cd "$REPO"
```

```bash
# file: launchers/train.sh
#!/usr/bin/env bash
# Training on the bot 8gpu class. Overrides: RUN_NAME, MODEL, NTOK, LEVELS, STEPS, NPROC, BATCH, EXTRA.
source "$(dirname "$0")/common.sh"
NPROC="${NPROC:-8}"
RUN_NAME="${RUN_NAME:-hier-fsq-n10-v2048}"
MODEL="${MODEL:-hier_fsq}"
STEPS="${STEPS:-300000}"
export OMP_NUM_THREADS=4
OVERRIDES=(run_name="$RUN_NAME" model="$MODEL" train.steps="$STEPS" train.batch_size="${BATCH:-256}"
           train.num_workers="${WORKERS:-10}" train.eval_every="${EVAL_EVERY:-10000}" train.ckpt_every="${CKPT_EVERY:-5000}")
[ -n "$NTOK" ] && OVERRIDES+=(model.num_tokens="$NTOK")
[ -n "$LEVELS" ] && OVERRIDES+=(model.quantizer.levels="$LEVELS")
[ -n "$EXTRA" ] && OVERRIDES+=(${EXTRA//;/ })
echo "[launch] ${OVERRIDES[*]}"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader || true
accelerate launch --num_machines 1 --num_processes "$NPROC" --multi_gpu --mixed_precision bf16 --dynamo_backend no \
  train.py "${OVERRIDES[@]}"
echo "[launch] done $(date -Is)"
```

```bash
# file: launchers/smoke.sh
#!/usr/bin/env bash
# Short end-to-end check on the IB node (tmux): pytest, then 200 training steps on 2 GPUs with Comet enabled.
source "$(dirname "$0")/common.sh"
echo "[smoke] pytest"
python -m pytest -q
echo "[smoke] training 200 steps on GPUs ${SMOKE_GPUS:-1,2}"
CUDA_VISIBLE_DEVICES="${SMOKE_GPUS:-1,2}" NPROC=2 STEPS=200 EVAL_EVERY=100 CKPT_EVERY=100 WORKERS=8 \
  RUN_NAME="${RUN_NAME:-hier-smoke}" bash --noprofile --norc "$(dirname "$0")/train.sh"
echo "[smoke] done $(date -Is)"
```

```bash
# file: launchers/submit.sh
#!/usr/bin/env bash
# Submit a training job to the bot queue (run on an EXP node from the directory holding CLOUD_USER_TOKEN).
# Usage: TAG=r09 RUN_NAME=hier-fsq-n10-v2048 NTOK=10 LEVELS='[8,8,8,4]' submit.sh
set -e -o pipefail
BASE=/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/projects/action_tokenization
SRC=$BASE/vqvae_lattent_actions/launchers
TAG="${TAG:?set TAG, e.g. r09}"
RUN_NAME="${RUN_NAME:-hier-fsq-n10-v2048}"
DIR=$BASE/runs/${TAG}_hier
cd /mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov
mkdir -p "$DIR"
cp "$SRC/common.sh" "$SRC/train.sh" "$DIR/"
rm -f "$DIR/exit_code"
ENVS="RUN_NAME=$RUN_NAME MODEL=${MODEL:-hier_fsq} NPROC=8 STEPS=${STEPS:-300000} WORKERS=${WORKERS:-10}"
[ -n "$NTOK" ] && ENVS="$ENVS NTOK=$NTOK"
[ -n "$LEVELS" ] && ENVS="$ENVS LEVELS=$LEVELS"
CMD="cd $DIR && env $ENVS bash --noprofile --norc $DIR/train.sh"
echo "[submit] lerobot-research-${TAG}-hier: $CMD"
bot submit -t 8gpu -H 48 -n "lerobot-research-${TAG}-hier" -c "$CMD" --json | tee "$DIR/submit.json"
```

```bash
# file: launchers/comet_upload.sh
#!/usr/bin/env bash
# Upload Comet offline archives (written when compute nodes have no internet) from a node that does.
source "$(dirname "$0")/common.sh"
DIR="${1:?usage: comet_upload.sh <directory with *.zip>}"
for archive in "$DIR"/*.zip; do
  echo "[comet] uploading $(basename "$archive")"
  python -m comet_ml.scripts.comet_upload "$archive"
done
```

```markdown
# file: README.md
# Hierarchical action tokenizer (unified action space)

Discrete tokenizer for VLA action chunks: a `T x 119` chunk of the unified `bimanual_rotation6d` space plus its
`{0,1}` padding mask becomes `N` tokens from a vocabulary of `V` codes. `T`, `N` and `V` are independent settings;
the model sees padding and body-part structure explicitly instead of guessing from zeros, and it never sees the
robot state.

## How it works

1. **Pointwise embedding.** Every `(timestep, dimension)` becomes a vector: real values through a shared scalar
   projector plus a per-dimension embedding, padded entries through one learned `[PAD]` vector; a group embedding
   (17 semantic groups: arms, hands, head, torso, base, legs) and a time embedding are added on top.
2. **Per-step queries.** `K = 17 + free` queries attend to the 119 positions of their timestep. Group queries are
   restricted by an attention mask to their own group, so a missing group produces a "group absent" token; free
   queries see everything.
3. **Dynamics.** Self-attention over all `T*K` tokens with a time code: the tokenizer models the trajectory, not
   isolated points.
4. **Latents.** `N` learned queries compress the sequence through Perceiver blocks; the quantizer (FSQ by default,
   VQ-EMA / LFQ / Gumbel available) turns them into `N` tokens.
5. **Decoder.** Mirrors the encoder, is conditioned on the same mask, and emits exactly zeros where the mask says
   padding; per-dimension heads read their group token plus a pooled summary of the free tokens.

Loss: masked MSE over real entries only (optional per-group weights) plus the quantizer's auxiliary term.

## Data

Chunks come from the `action_chunks` package (mixture R0_v2.1, 95 embodiments, manifest `r0_v2.1-6885099`):
`WeightedChunkDataset` with the VLA mixture weights for training, and a frozen eval set of 44,642 held-out chunks
(<= 500 per embodiment) shared with the FAST / BEAST / ActionCodec / OAT arms, so every number is comparable.

## Running

```bash
python train.py run_name=hier-fsq-n10-v2048                    # defaults: N=10, V=2048 (FSQ [8,8,8,4]), 300k steps
python train.py model=hier_vq model.num_tokens=16 train.steps=100000
accelerate launch --multi_gpu --num_processes 8 train.py run_name=... # multi-GPU
```

On the cluster use `launchers/`: `smoke.sh` (IB, tmux, 200 steps on 2 GPUs), `submit.sh` (bot `8gpu` queue),
`comet_upload.sh` (ship offline Comet archives). `launchers/common.sh` puts `action_chunks`, `pylib` and
`pylib_comet` on `PYTHONPATH` and points `COMET_CONFIG` at the key file outside the repository.

## Experiment tracking

Comet ML, workspace `dont4rootme`, project `hier-action-tokenizer`. `comet.mode=auto` logs online when the API is
reachable and writes an offline archive otherwise; every metric is also mirrored into `train_log.jsonl` inside the
run directory. The API key is never stored in the repository: it comes from `COMET_API_KEY` or `COMET_CONFIG`
(`~/.comet.config` locally, `/mnt/.../afedorov/.comet.config` on the cluster).

## Tests

```bash
PYTHONPATH=.:../action_chunks pytest -q
```

Design: `docs/superpowers/specs/2026-09-11-hierarchical-action-tokenizer-design.md`.
Implementation plan: `docs/superpowers/plans/2026-09-12-hier-tokenizer-implementation.md`. Run journal: `docs/runs.md`.
```

```markdown
# file: docs/runs.md
# Журнал запусков

`$BASE` = `/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/projects/action_tokenization`, репозиторий на кластере
`$BASE/vqvae_lattent_actions`, данные `action_chunks` (манифест `r0_v2.1-6885099`, sha 05d18b2b), общий eval-набор
`$BASE/runs/eval_set/eval_set-05d18b2b-val500.npz` (44 642 чанка). Трекинг: Comet, workspace `dont4rootme`,
проект `hier-action-tokenizer`.

Точки отсчёта на том же eval-наборе: FAST+ 59 токенов, RMSE 0.0194; BEAST nb6_deg2 149 токенов, RMSE 0.0300;
ActionCodec 10 токенов, RMSE 0.0806; OAT 10 токенов, RMSE 0.1028.

| Имя | Где | Что | Каталог | Статус |
|---|---|---|---|---|
```
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=.:/Users/artemon/projects/action_chunks /Users/artemon/projects/action_chunks/.venv/bin/pytest -q`
Expected: whole suite green

- [ ] **Step 5: Delete the old launcher, commit and push the branch**

```bash
git rm -q train.sh 2>/dev/null || true
git add -A
git commit -m "entry point, hydra configs, cluster launchers and docs"
git push -u origin feat/hier-tokenizer
```

---

### Task 8: Cluster verification and the first training run

**Files:** none in the repository (run directories under `$BASE/runs/`).

- [ ] **Step 1: Put the repository on the cluster**

```bash
ssh diffusion-h100-9-ib 'BASE=/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/projects/action_tokenization; \
  git -C $BASE clone -q https://github.com/Dont4rootMe/vqvae_lattent_actions.git 2>/dev/null; \
  git -C $BASE/vqvae_lattent_actions fetch -q origin && git -C $BASE/vqvae_lattent_actions checkout -q feat/hier-tokenizer && \
  git -C $BASE/vqvae_lattent_actions pull -q && git -C $BASE/vqvae_lattent_actions rev-parse --short HEAD'
```

- [ ] **Step 2: Run the suite on the cluster** (same environment as training)

```bash
ssh diffusion-h100-9-ib 'bash --noprofile --norc -lc "cd <REPO> && source launchers/common.sh && python -m pytest -q"'
```
Expected: green; `action_chunks` and `comet_ml` import from `PYTHONPATH`.

- [ ] **Step 3: Smoke on IB in tmux** (2 free GPUs, 200 steps, real R0_v2.1 data)

```bash
ssh diffusion-h100-9-ib 'BASE=...; RUN=$BASE/runs/hier_smoke; mkdir -p $RUN; \
  cp $BASE/vqvae_lattent_actions/launchers/{common.sh,train.sh,smoke.sh} $RUN/; \
  tmux new-session -d -s lerobot-research-r09-smoke "bash --noprofile --norc $RUN/smoke.sh >> $RUN/run.log 2>&1"'
```
Success criteria in `run.log`: pytest green; `[comet] mode=online|offline`; model parameter count and
`10 tokens x 2048 codes = 110 bits/chunk`; steps advancing with `it/s`; `eval step 100/200` lines with
`rmse`, `codes`, `perplexity`; `padding_invariance_mismatch = 0.0` in `eval_step*.json`; `final export`; `exit=0`.

- [ ] **Step 4: Submit the first production run to the bot queue**

```bash
ssh diffusion-1-exp 'cd /mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov && \
  TAG=r09 RUN_NAME=hier-fsq-n10-v2048 NTOK=10 LEVELS="[8,8,8,4]" \
  bash /mnt/.../vqvae_lattent_actions/launchers/submit.sh'
```
Then verify with `bot info <id> --json` that it is running, and that the log shows 8 ranks and advancing steps.

- [ ] **Step 5: Record the run and checkpoint memory**

- add the row to `docs/runs.md` (name, task id, config, directory, status) and commit;
- `memory_checkpoint` in topic `hierarchical-unified-action-tokenizer`: implementation finished, smoke result,
  job id, where the reports land, what the first eval numbers are compared with ActionCodec 0.081.

---

## Self-Review

**Spec coverage.** Unified 119 layout from the fork's yaml (T1) — spec §2; padding and group structure inside the
model (T3) — §3.1; T/N/V independent with V <= 2048 (T3, configs in T7) — §1; no state, mask in both directions (T3) — §3.3;
quantizers incl. FSQ default (T2) — §3.2; masked loss with optional group weights (T3) — §3.4; data and shared eval
set (T4) — §2; metrics compatible with the arm table plus usage and padding invariance (T5) — §5; accelerate training
with the scheduler fix, resume, export (T6) — §4; repo structure and deletions (T1-T7) — §6; the seven tests of §7 map to
`test_model.py` (padding invariance, zeros, vocabulary, shapes, group masking), `test_quantizers.py` (vocabulary),
`test_loop.py` (short training with resume), `test_metrics.py` (equality with `tokenizer_arms.metrics`).
Deviation from the spec, deliberate: tracking is Comet instead of tensorboard (user's decision of 2026-09-12), and the
scalar projector defaults to a small MLP rather than a bare linear map (`value_embedding` switches back).

**Placeholders.** None: every step carries the full file content or the exact command.

**Type consistency.** `QuantizerOutput(codes, indices, aux_loss)` is produced in T2 and consumed in T3;
`quantizer.code_dim` feeds `to_code`, `quantizer.out_dim` feeds `from_code`; `batch_to_inputs` returns
`(actions, mask, embodiment_index)` in T4 and is used in T6; `EvalSet.batches()` yields
`(actions, valid, time_valid, embodiment_index)` in T4 and is consumed in T5; `evaluate_tokenizer` returns the
`MetricAccumulator.summary()` shape (`total`/`per_embodiment`) plus `usage`, which `write_report` accepts.
