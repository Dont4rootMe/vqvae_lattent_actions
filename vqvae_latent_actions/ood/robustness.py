"""Does the tokenizer still represent a chunk once it leaves the training distribution, and does it do so smoothly?

Two questions, measured separately:

* **Fidelity.** Reconstruction error on the perturbed chunk itself. A tokenizer that silently snaps an unfamiliar
  motion back onto a familiar one scores well on the clean set and badly here.
* **Smoothness.** How the tokens and the decoded output move when the input moves a little. A VLA has to learn and
  generalize over these tokens, so a small change of the action must not scatter them at random.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..data.chunks import EvalSet
from ..layout import UnifiedLayout
from ..training.metrics import masked_errors
from . import perturb


def _device(model, device: Any) -> torch.device:
    return torch.device(device) if device is not None else next(model.parameters()).device


@torch.no_grad()
def encode_decode(model, actions: np.ndarray, valid: np.ndarray, *, batch_size: int = 512,
                  device: Any = None) -> tuple[np.ndarray, np.ndarray]:
    """Tokens [N, K] and reconstruction [N, T, D], both as numpy."""
    dev = _device(model, device)
    tokens, recon = [], []
    for start in range(0, len(actions), batch_size):
        sl = slice(start, start + batch_size)
        a = torch.as_tensor(actions[sl], device=dev)
        m = torch.as_tensor(valid[sl], device=dev)
        t, r = model.encode_decode(a, m)
        tokens.append(t.cpu().numpy())
        recon.append(r.float().cpu().numpy())
    return np.concatenate(tokens), np.concatenate(recon)


def reconstruction(actions: np.ndarray, recon: np.ndarray, valid: np.ndarray) -> dict:
    e = masked_errors(actions, recon, valid)
    return {"rmse": float(np.sqrt(e["mse"].mean())), "l1": float(e["l1"].mean()),
            "max_abs": float(e["max_abs"].mean())}


def token_agreement(clean: np.ndarray, perturbed: np.ndarray) -> dict:
    """How much of the token string survives: whole chunks unchanged, and positions unchanged."""
    same = np.asarray(clean) == np.asarray(perturbed)
    return {"tokens_identical": float(same.all(axis=-1).mean()), "token_agreement": float(same.mean())}


def _row(model, eval_set: EvalSet, out: perturb.Perturbed, clean_tokens: np.ndarray, clean_recon: np.ndarray,
         clean: dict, *, batch_size: int, device: Any) -> dict:
    tokens, recon = encode_decode(model, out.actions, out.valid, batch_size=batch_size, device=device)
    shared = out.valid & eval_set.valid
    return {**reconstruction(out.actions, recon, out.valid), **token_agreement(clean_tokens, tokens),
            "clean_rmse": clean["rmse"],
            # how far the decoded chunk moves: a graceful tokenizer moves its output about as far as its input
            "input_shift": float(np.sqrt(masked_errors(eval_set.actions, out.actions, shared)["mse"].mean())),
            "output_shift": float(np.sqrt(masked_errors(clean_recon, recon, shared)["mse"].mean()))}


def _clean_pass(model, eval_set: EvalSet, *, batch_size: int, device: Any):
    tokens, recon = encode_decode(model, eval_set.actions, eval_set.valid, batch_size=batch_size, device=device)
    return tokens, recon, reconstruction(eval_set.actions, recon, eval_set.valid)


def noise_sweep(model, eval_set: EvalSet, *, sigmas=(0.0, 0.01, 0.05, 0.1, 0.25), batch_size: int = 512,
                device: Any = None, seed: int = 0) -> list[dict]:
    clean_tokens, clean_recon, clean = _clean_pass(model, eval_set, batch_size=batch_size, device=device)
    rows = []
    for sigma in sigmas:
        out = perturb.gaussian_noise(eval_set.actions, eval_set.valid, sigma=float(sigma), seed=seed)
        rows.append({"sigma": float(sigma),
                     **_row(model, eval_set, out, clean_tokens, clean_recon, clean, batch_size=batch_size,
                            device=device)})
    return rows


def amplitude_sweep(model, eval_set: EvalSet, *, factors=(0.5, 1.5, 2.0, 4.0), batch_size: int = 512,
                    device: Any = None) -> list[dict]:
    clean_tokens, clean_recon, clean = _clean_pass(model, eval_set, batch_size=batch_size, device=device)
    rows = []
    for factor in factors:
        out = perturb.amplitude_scale(eval_set.actions, eval_set.valid, factor=float(factor))
        rows.append({"factor": float(factor),
                     **_row(model, eval_set, out, clean_tokens, clean_recon, clean, batch_size=batch_size,
                            device=device)})
    return rows


def time_sweep(model, eval_set: EvalSet, *, factors=(0.5, 2.0), batch_size: int = 512,
               device: Any = None) -> list[dict]:
    clean_tokens, clean_recon, clean = _clean_pass(model, eval_set, batch_size=batch_size, device=device)
    rows = []
    for factor in factors:
        out = perturb.time_stretch(eval_set.actions, eval_set.valid, factor=float(factor))
        rows.append({"factor": float(factor),
                     **_row(model, eval_set, out, clean_tokens, clean_recon, clean, batch_size=batch_size,
                            device=device)})
    return rows


def group_dropout(model, eval_set: EvalSet, layout: UnifiedLayout, *, batch_size: int = 512,
                  device: Any = None) -> list[dict]:
    """Drop one body part at a time: an unfamiliar combination of present groups, with the rest unchanged."""
    clean_tokens, clean_recon, clean = _clean_pass(model, eval_set, batch_size=batch_size, device=device)
    rows = []
    for name in layout.names:
        start, end = layout.intervals()[name]
        present = float(eval_set.valid[:, :, start:end].any(axis=(1, 2)).mean())
        if present == 0.0:                 # the group is absent everywhere; dropping it changes nothing
            continue
        out = perturb.drop_group(eval_set.actions, eval_set.valid, layout, name)
        rows.append({"group": name, "present_fraction": present,
                     **_row(model, eval_set, out, clean_tokens, clean_recon, clean, batch_size=batch_size,
                            device=device)})
    return rows


@torch.no_grad()
def latent_interpolation(model, eval_set: EvalSet, *, num_pairs: int = 256, alphas=(0.0, 0.25, 0.5, 0.75, 1.0),
                         seed: int = 0, device: Any = None) -> dict:
    """Walk the latent line between two chunks and watch the decoded output.

    A usable embedding space moves the output steadily from one chunk to the other: the decoded midpoint sits about
    halfway (`midpoint_ratio` near 0.5) and the distance to the starting chunk never turns back (`monotone_fraction`
    near 1). A space that jumps instead is one a VLA has to memorize rather than generalize over.
    """
    dev = _device(model, device)
    rng = np.random.default_rng(seed)
    count = min(num_pairs, len(eval_set) // 2)
    picks = rng.choice(len(eval_set), size=2 * count, replace=False)
    left, right = picks[:count], picks[count:]
    mask = torch.as_tensor(eval_set.valid[left] & eval_set.valid[right], device=dev)
    a = torch.as_tensor(eval_set.actions[left], device=dev).masked_fill(~mask, 0.0)
    b = torch.as_tensor(eval_set.actions[right], device=dev).masked_fill(~mask, 0.0)
    za, zb = model.encode_continuous(a, mask), model.encode_continuous(b, mask)
    weights = mask.float()

    def distance(x, y):
        return (((x - y) ** 2) * weights).sum(dim=(1, 2)).div(weights.sum(dim=(1, 2)).clamp_min(1.0)).sqrt()

    decoded = [model.decode_latents(model.quantizer.bound(za * (1 - t) + zb * t), mask) for t in alphas]
    to_start = torch.stack([distance(d, decoded[0]) for d in decoded])          # [A, count]
    span = to_start[-1].clamp_min(1e-8)
    steps = to_start[1:] - to_start[:-1]
    midpoint = to_start[len(alphas) // 2] / span
    return {"num_pairs": int(count), "alphas": [float(t) for t in alphas],
            "midpoint_ratio": float(midpoint.mean()),
            "monotone_fraction": float((steps >= -1e-6).all(dim=0).float().mean()),
            "travel": [float(v) for v in to_start.mean(dim=1)],
            "endpoint_distance": float(to_start[-1].mean())}


__all__ = ["amplitude_sweep", "encode_decode", "group_dropout", "latent_interpolation", "noise_sweep",
           "reconstruction", "time_sweep", "token_agreement"]
