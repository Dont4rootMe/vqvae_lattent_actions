"""Out-of-distribution transforms of a chunk: what a VLA will hand the tokenizer once it leaves the training set.

Every transform keeps the shape, keeps padded entries exactly zero and masked, and touches real entries only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..layout import UnifiedLayout


@dataclass(frozen=True)
class Perturbed:
    actions: np.ndarray   # [N, T, D] float32, zeros where the mask says padding
    valid: np.ndarray     # [N, T, D] bool


def _clean(actions: np.ndarray, valid: np.ndarray) -> Perturbed:
    out = np.asarray(actions, dtype=np.float32).copy()
    out[~valid] = 0.0
    return Perturbed(actions=out, valid=valid)


def gaussian_noise(actions: np.ndarray, valid: np.ndarray, *, sigma: float, seed: int = 0) -> Perturbed:
    """Sensor-grade jitter on the real entries; `sigma` is in units of the normalized action space."""
    if sigma == 0.0:
        return _clean(actions, valid)
    rng = np.random.default_rng(seed)
    noise = rng.normal(scale=sigma, size=actions.shape).astype(np.float32)
    return _clean(np.asarray(actions, dtype=np.float32) + noise * valid, valid)


def amplitude_scale(actions: np.ndarray, valid: np.ndarray, *, factor: float) -> Perturbed:
    """A motion of unfamiliar size: `factor > 1` pushes values past the range the tokenizer was trained on."""
    return _clean(np.asarray(actions, dtype=np.float32) * factor, valid)


def constant_offset(actions: np.ndarray, valid: np.ndarray, *, delta: float) -> Perturbed:
    """A shifted workspace: the same trajectory around a different operating point."""
    return _clean(np.asarray(actions, dtype=np.float32) + delta, valid)


def time_stretch(actions: np.ndarray, valid: np.ndarray, *, factor: float) -> Perturbed:
    """Resample the horizon: `factor > 1` replays the chunk faster, `< 1` slower, both by linear interpolation.

    Steps that would fall past the end of the chunk hold its last step, so a faster replay never invents motion.
    """
    actions = np.asarray(actions, dtype=np.float32)
    horizon = actions.shape[1]
    source = np.clip(np.arange(horizon, dtype=np.float32) * float(factor), 0.0, horizon - 1)
    low, high = np.floor(source).astype(int), np.ceil(source).astype(int)
    weight = (source - low).astype(np.float32).reshape(1, -1, 1)
    out = actions[:, low] * (1.0 - weight) + actions[:, high] * weight
    mask = valid[:, low] & valid[:, high]
    return _clean(out, mask)


def drop_group(actions: np.ndarray, valid: np.ndarray, layout: UnifiedLayout, group: str) -> Perturbed:
    """Turn one body part off: a combination of present groups the tokenizer may never have seen."""
    start, end = layout.intervals()[group]
    mask = valid.copy()
    mask[:, :, start:end] = False
    return _clean(actions, mask)


def keep_only_groups(actions: np.ndarray, valid: np.ndarray, layout: UnifiedLayout, groups: list[str]) -> Perturbed:
    """The mirror image of `drop_group`: only the named groups stay present."""
    mask = np.zeros_like(valid)
    for name in groups:
        start, end = layout.intervals()[name]
        mask[:, :, start:end] = valid[:, :, start:end]
    return _clean(actions, mask)


__all__ = ["Perturbed", "amplitude_scale", "constant_offset", "drop_group", "gaussian_noise", "keep_only_groups",
           "time_stretch"]
