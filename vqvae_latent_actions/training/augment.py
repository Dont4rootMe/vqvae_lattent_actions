"""Training-time augmentation aimed at what the OOD suite found the tokenizer cannot handle.

Every transform is applied per chunk with its own probability and changes the input and the reconstruction target
alike, so the model learns to represent a wider distribution rather than to denoise:

* amplitude: the same motion at a different size (log-uniform factor). Scaling by 0.5 or 1.5 cost 5-7x the error.
* noise: small gaussian jitter on the real entries.
* group drop: one present body part switched off, so each part has to be encoded on its own.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor


@dataclass(frozen=True)
class AugmentConfig:
    amplitude_prob: float = 0.0
    amplitude_range: tuple[float, float] = (0.5, 1.5)
    noise_prob: float = 0.0
    noise_sigma: float = 0.02           # upper bound; each chunk draws its own sigma in [0, noise_sigma]
    group_drop_prob: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.amplitude_prob > 0 or self.noise_prob > 0 or self.group_drop_prob > 0

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "AugmentConfig":
        payload = dict(payload or {})
        if "amplitude_range" in payload:
            payload["amplitude_range"] = tuple(float(v) for v in payload["amplitude_range"])
        return cls(**payload)


def present_groups(mask: Tensor, membership: Tensor) -> Tensor:
    """[B, G] bool: the group has at least one real entry anywhere in the chunk."""
    any_dim = mask.any(dim=1).float()                                     # [B, D]
    return (any_dim @ membership.to(any_dim.device, any_dim.dtype).T) > 0


def _uniform(shape, generator, device) -> Tensor:
    return torch.rand(shape, generator=generator, device=device)


def augment(actions: Tensor, mask: Tensor, membership: Tensor, cfg: AugmentConfig,
            generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
    """New (actions, mask); padded entries stay exactly zero and the mask never gains an entry."""
    if not cfg.enabled:
        return actions, mask
    b, device = actions.shape[0], actions.device
    # draw on the generator's device, then move: a CPU generator keeps the draws reproducible across devices
    gen_device = generator.device if generator is not None else device
    x, m = actions, mask
    if cfg.amplitude_prob > 0:
        lo, hi = (math.log(v) for v in cfg.amplitude_range)
        factor = torch.exp(lo + (hi - lo) * _uniform(b, generator, gen_device)).to(device)
        apply = (_uniform(b, generator, gen_device) < cfg.amplitude_prob).to(device)
        x = x * torch.where(apply, factor, torch.ones_like(factor)).view(b, 1, 1).to(x.dtype)
    if cfg.noise_prob > 0:
        sigma = (cfg.noise_sigma * _uniform(b, generator, gen_device)).to(device)
        apply = (_uniform(b, generator, gen_device) < cfg.noise_prob).to(device)
        sigma = torch.where(apply, sigma, torch.zeros_like(sigma)).view(b, 1, 1)
        noise = torch.randn(x.shape, generator=generator, device=gen_device).to(device, x.dtype)
        x = x + noise * sigma.to(x.dtype)
    if cfg.group_drop_prob > 0:
        membership = membership.to(device)
        present = present_groups(m, membership)                              # [B, G]
        eligible = (_uniform(b, generator, gen_device) < cfg.group_drop_prob).to(device) & (present.sum(-1) >= 2)
        scores = _uniform(present.shape, generator, gen_device).to(device).masked_fill(~present, -1.0)
        dropped = membership[scores.argmax(dim=-1)] & eligible.unsqueeze(-1)  # [B, D]
        m = m & ~dropped.unsqueeze(1)
    return x.masked_fill(~m, 0.0), m


__all__ = ["AugmentConfig", "augment", "present_groups"]
