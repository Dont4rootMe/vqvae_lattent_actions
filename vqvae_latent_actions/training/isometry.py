"""Relaxed distortion measure: keep the decoder's Jacobian isotropic, so the latent space is smooth.

From `Regularized Autoencoders for Isometric Representation Learning` (Lee, Yoon, Son, Park, ICLR 2022), reused in
`Isometric Representation Learning for Disentangled Latent Space of Diffusion Models` (ICML 2024). For the pullback
metric `G = J^T H J` of the decoder `z -> x_hat` the penalty is

    RDM = E[Tr(G^2)] / E[Tr(G)]^2,

which is the inverse participation ratio of the eigenvalues of `G`. It is minimal (`1/m`, `m = dim z`) exactly when
`G` is a multiple of the identity, and it does not change when the decoder is scaled, so it asks for conformality,
not for a fixed scale: the codebook keeps whatever norm it settles on. Its reciprocal, the participation ratio, is
the readable version — the number of latent directions the decoder actually uses.

Both traces come from probes: with `v, w ~ N(0, I)` independent,

    Tr(G)   = E[ (Jv)^T H (Jv) ],        Tr(G^2) = E[ ((Jv)^T H (Jw))^2 ].

`Jv` is taken by central differences instead of a JVP: the estimate then contains no second derivative, so the
training step stays an ordinary single backward that DDP can reduce, at the cost of four extra decoder passes over
a small sub-batch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch
from torch import Tensor


@dataclass(frozen=True)
class IsometryConfig:
    weight: float = 0.0             # 0 disables the penalty entirely
    subbatch: int = 64              # chunks probed per step; the estimate is averaged over them
    epsilon: float = 0.01           # finite-difference step, relative to the RMS of the latent
    alpha_jitter: float = 0.2       # probe at a*z + (1-a)*z[perm], a ~ U(-jitter, 1+jitter), as in the paper
    warmup_steps: int = 0           # let the reconstruction settle first
    every: int = 1                  # apply every N steps (the weight is scaled up to keep the average pull)

    @property
    def enabled(self) -> bool:
        return self.weight > 0.0

    def active(self, step: int) -> bool:
        return self.enabled and step >= self.warmup_steps and (self.every <= 1 or step % self.every == 0)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "IsometryConfig":
        return cls(**dict(payload or {}))


def _probe_points(z: Tensor, cfg: IsometryConfig, generator: torch.Generator | None) -> Tensor:
    """A sub-batch of latents, mixed with a shuffled copy so the metric is measured between samples too."""
    z = z[: cfg.subbatch].detach()
    if cfg.alpha_jitter <= 0.0:
        return z
    b = z.shape[0]
    gen_device = generator.device if generator is not None else z.device
    perm = torch.randperm(b, generator=generator, device=gen_device).to(z.device)
    lo, hi = -cfg.alpha_jitter, 1.0 + cfg.alpha_jitter
    alpha = (lo + (hi - lo) * torch.rand(b, generator=generator, device=gen_device)).to(z.device, z.dtype)
    shape = (b,) + (1,) * (z.dim() - 1)
    return alpha.view(shape) * z + (1.0 - alpha.view(shape)) * z[perm]


def isometry_penalty(decode: Callable[[Tensor], Tensor], latents: Tensor, weights: Tensor | None,
                     cfg: IsometryConfig, generator: torch.Generator | None = None) -> dict | None:
    """`None` when disabled; otherwise the weighted loss plus the numbers worth logging.

    `weights` is the diagonal of `H`, broadcastable to the decoder's output: the mask, so padded entries carry no
    geometry, times whatever per-dimension weights the reconstruction loss uses.
    """
    if not cfg.enabled:
        return None
    # bf16 has 7 mantissa bits: a central difference over a step of 1% would be pure rounding noise, and noise looks
    # isotropic, so the penalty would report a perfect geometry and pull on nothing. The probes run in fp32.
    device_type = latents.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        return _penalty(decode, latents.float(), None if weights is None else weights.float(), cfg, generator)


def _penalty(decode, latents: Tensor, weights: Tensor | None, cfg: IsometryConfig,
             generator: torch.Generator | None) -> dict:
    z = _probe_points(latents, cfg, generator)
    gen_device = generator.device if generator is not None else z.device
    scale = z.detach().pow(2).mean().sqrt().clamp_min(1e-3)
    eps = cfg.epsilon * scale

    def jacobian_product(direction: Tensor) -> Tensor:
        return (decode(z + eps * direction) - decode(z - eps * direction)) / (2 * eps)

    v, w = (torch.randn(z.shape, generator=generator, device=gen_device).to(z.device, z.dtype) for _ in range(2))
    jv, jw = jacobian_product(v).float(), jacobian_product(w).float()
    h = 1.0 if weights is None else weights[: z.shape[0]].to(jv.dtype)
    dims = tuple(range(1, jv.dim()))
    trace = (h * jv * jv).sum(dim=dims)                      # [B]: v^T G v, an unbiased draw of Tr(G)
    cross = (h * jv * jw).sum(dim=dims)                      # [B]: v^T G w, whose square averages to Tr(G^2)
    mean_trace = trace.mean()
    rdm = (cross ** 2).mean() / mean_trace.clamp_min(1e-12) ** 2
    return {"loss": cfg.weight * rdm, "rdm": rdm.detach(), "trace": float(mean_trace.detach()),
            "trace_squared": float((cross ** 2).mean().detach()),
            "participation_ratio": float(1.0 / rdm.detach().clamp_min(1e-12))}


__all__ = ["IsometryConfig", "isometry_penalty"]
