"""Quantizers: N continuous latents -> N discrete tokens, behind one interface.

FSQ is the default (no codebook collapse, deterministic, vocabulary = product of levels). VQ-EMA covers
arbitrary vocabulary sizes, LFQ and Gumbel-softmax are kept from the earlier experiments in this repo.
"""
from __future__ import annotations

import inspect
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

    def bound(self, z: Tensor) -> Tensor:
        """The value range the decoder sees once quantization is on, without the rounding.

        Identity for a learned codebook on the euclidean path: it follows whatever scale the encoder settles on.
        FSQ overrides it because its grid is fixed, and a cosine codebook overrides it because it compares
        directions.
        """
        return z

    def saturation(self, z: Tensor) -> float:
        """Fraction of the code that sits where the quantizer can no longer resolve it. 0 unless bounded."""
        return 0.0

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

    def bound(self, z: Tensor) -> Tensor:
        return self._bound(z.float()) / self._half_width

    def saturation(self, z: Tensor) -> float:
        inner = torch.tanh(z.float() + self._shift)
        return float((inner.abs() > 0.99).float().mean())

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
                 eps: float = 1e-5, dead_threshold: float = 1.0, kmeans_init: bool = True,
                 kmeans_iters: int = 10, restart_ratio: float = 0.1, seed_samples_per_code: int = 4,
                 cosine: bool = False, squash: bool = False) -> None:
        super().__init__()
        self.vocab_size, self.code_dim, self.out_dim = int(vocab_size), int(code_dim), int(code_dim)
        self.commitment, self.decay, self.eps, self.dead_threshold = float(commitment), float(decay), float(eps), float(dead_threshold)
        self.kmeans_init, self.kmeans_iters = bool(kmeans_init), int(kmeans_iters)
        self.restart_ratio = float(restart_ratio)   # codes used far less than average are restarted
        self.seed_samples_per_code = int(seed_samples_per_code)
        # Nothing anchors the scale of a latent that a learned codebook consumes. Measured at 10 tokens: the code
        # norm grew from 1.0 to 9.3 between steps 25k and 40k and three quarters of the codes fell out of use.
        # Two anchors: `cosine` compares directions and drops magnitude; `squash` passes the code through tanh,
        # which bounds it to (-1, 1) but keeps magnitude. EMA means of squashed codes stay inside the same box.
        if cosine and squash:
            raise ValueError("cosine and squash are alternative ways to anchor the code scale; pick one")
        self.cosine, self.squash = bool(cosine), bool(squash)
        codebook = torch.randn(self.vocab_size, self.code_dim) * 0.1
        self.register_buffer("codebook", codebook)
        self.register_buffer("cluster_size", torch.ones(self.vocab_size))
        self.register_buffer("embed_sum", codebook.clone())
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))
        self._seed_buffer: list[Tensor] = []      # latents collected until there are enough to seed the codebook

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
    def _seed_plus_plus(self, flat: Tensor) -> Tensor:
        """k-means++ seeding: each new code is drawn far from the ones already chosen.

        Uniform seeding leaves whole clusters uncovered, and EMA cannot escape that afterwards (duplicated codes
        keep enough usage to never look dead).
        """
        first = torch.randint(0, flat.shape[0], (1,), device=flat.device)
        codes = [flat[first]]
        nearest = (flat - codes[0]).pow(2).sum(dim=1)
        for _ in range(self.vocab_size - 1):
            total = nearest.sum()
            if float(total) <= 0:
                index = torch.randint(0, flat.shape[0], (1,), device=flat.device)
            else:
                index = torch.multinomial(nearest / total, 1)
            chosen = flat[index]
            codes.append(chosen)
            nearest = torch.minimum(nearest, (flat - chosen).pow(2).sum(dim=1))
        return torch.cat(codes, dim=0)

    @torch.no_grad()
    def _lloyd(self, flat: Tensor, codes: Tensor) -> Tensor:
        for _ in range(self.kmeans_iters):
            assignment = torch.cdist(flat, codes).argmin(dim=1)
            onehot = F.one_hot(assignment, self.vocab_size).to(flat.dtype)
            counts = onehot.sum(0)
            moved = (onehot.t() @ flat) / counts.clamp_min(1.0).unsqueeze(1)
            codes = torch.where(counts.unsqueeze(1) > 0, moved, codes)
        return codes

    @torch.no_grad()
    def _collect(self, flat: Tensor) -> Tensor | None:
        """Hold latents back until one step's worth is enough to seed every code.

        A single batch can carry fewer vectors than the codebook has entries (256 chunks x 5 tokens against 2048
        codes), and seeding from it draws the same vector many times. The duplicates then survive, because a code
        that shares its position with others still collects enough usage to never look dead. Even an exactly
        sufficient sample is too thin: k-means++ needs several points per code to place one.
        """
        if not self.kmeans_init:
            return flat
        self._seed_buffer.append(flat)
        total = sum(part.shape[0] for part in self._seed_buffer)
        if total < self.seed_samples_per_code * self.vocab_size:
            return None
        collected = torch.cat(self._seed_buffer)
        self._seed_buffer = []
        return collected

    @torch.no_grad()
    def _initialize(self, flat: Tensor) -> None:
        if self.kmeans_init and flat.shape[0] >= self.vocab_size:
            codes = self._lloyd(flat, self._seed_plus_plus(flat))
        else:
            pick = torch.randint(0, flat.shape[0], (self.vocab_size,), device=flat.device)
            codes = flat[pick].clone()
        codes = self._broadcast(codes.contiguous())
        self.codebook.copy_(codes)
        self.embed_sum.copy_(codes)
        self.cluster_size.fill_(1.0)
        self.initialized.fill_(True)

    def _anchor(self, x: Tensor) -> Tensor:
        if self.cosine:
            return F.normalize(x, dim=-1)
        if self.squash:
            return torch.tanh(x)
        return x

    def bound(self, z: Tensor) -> Tensor:
        return self._anchor(z.float())

    def forward(self, z: Tensor) -> QuantizerOutput:
        # Autocast would run the matching matmul in bfloat16 even after .float(): 7 mantissa bits, so the nearest
        # code is often picked wrong (with cosine every similarity is close to 1), and fp32 evaluation would
        # disagree with training. The whole codebook step runs in fp32.
        with torch.autocast(device_type=z.device.type, enabled=False):
            return self._quantize(z.float())

    def _quantize(self, z: Tensor) -> QuantizerOutput:
        z = self._anchor(z)
        flat = z.reshape(-1, self.code_dim)
        if self.training and not bool(self.initialized):
            collected = self._collect(flat.detach())
            if collected is not None:
                self._initialize(collected)
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
                floor = torch.clamp(self.cluster_size.mean() * self.restart_ratio, min=self.dead_threshold)
                dead = self.cluster_size < floor
                # One sample may revive one code. Repeating samples to fill the quota writes the same vector into
                # many codes, which is the collapse the restart exists to prevent; with a batch smaller than the
                # codebook that happens on every step.
                count = min(int(dead.sum()), int(flat.shape[0]))
                if count:
                    scores = (-self.cluster_size).masked_fill(~dead, float("-inf"))
                    victims = scores.topk(count).indices                                # the emptiest codes
                    far = torch.cdist(flat.detach(), self.codebook).min(dim=1).values   # worst-covered samples
                    pick = far.topk(count).indices
                    replacement = self._broadcast(flat.detach()[pick].clone().contiguous())
                    self.codebook[victims] = replacement
                    self.embed_sum[victims] = replacement
                    self.cluster_size[victims] = 1.0
                if self.cosine:                       # a mean of unit vectors is shorter than one
                    self.codebook.copy_(F.normalize(self.codebook, dim=-1))
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


KINDS: Mapping[str, type] = {"fsq": FSQ, "vq": VQEMA, "vqema": VQEMA, "vq_ema": VQEMA,
                             "lfq": LFQWrapper, "gumbel": GumbelQuantizer}


def build_quantizer(config: Mapping[str, Any]) -> Quantizer:
    config = dict(config)
    kind = str(config.pop("type", "fsq")).lower()
    if kind not in KINDS:
        raise ValueError(f"unknown quantizer type {kind!r}, expected one of {sorted(KINDS)}")
    cls = KINDS[kind]
    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
    unexpected = sorted(set(config) - accepted)
    if unexpected:   # config groups merge, so a key from another quantizer rides along unnoticed
        raise ValueError(f"quantizer {kind!r} does not accept {unexpected}; it accepts {sorted(accepted)}")
    return cls(**config)


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
