"""Pre-LayerNorm attention blocks built on torch SDPA. Boolean masks: True means "may attend"."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0, qk_norm: bool = False) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")
        self.heads, self.dropout = int(heads), float(dropout)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, 2 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        # Nothing bounds q.k otherwise: over a 300k-step run one decoder attention grew its logits to 3e7 and went
        # one-hot, and a run diverged. Layer-normalized q and k keep the logits within sqrt(head_dim) times the gains.
        self.qk_norm = bool(qk_norm)
        self.q_norm = nn.LayerNorm(dim // heads) if self.qk_norm else None
        self.k_norm = nn.LayerNorm(dim // heads) if self.qk_norm else None

    def _qkv(self, x: Tensor, context: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        b, lq, dim = x.shape
        lk = context.shape[1]
        head_dim = dim // self.heads
        q = self.to_q(x).view(b, lq, self.heads, head_dim).transpose(1, 2)
        k, v = self.to_kv(context).view(b, lk, 2, self.heads, head_dim).permute(2, 0, 3, 1, 4)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        return q, k, v

    def forward(self, x: Tensor, context: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        b, lq, dim = x.shape
        q, k, v = self._qkv(x, context)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                             dropout_p=self.dropout if self.training else 0.0)
        return self.proj(out.transpose(1, 2).reshape(b, lq, dim))

    @torch.no_grad()
    def logits(self, x: Tensor, context: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        """The pre-softmax attention logits forward uses, in fp32; masked entries are -inf."""
        q, k, _ = self._qkv(x, context)
        scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
        return scores if attn_mask is None else scores.masked_fill(~attn_mask, float("-inf"))


class CrossBlock(nn.Module):
    """x attends to context."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0, qk_norm: bool = False) -> None:
        super().__init__()
        self.norm_q, self.norm_kv = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout, qk_norm)

    def forward(self, x: Tensor, context: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        return x + self.attn(self.norm_q(x), self.norm_kv(context), attn_mask)


class SelfBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0, qk_norm: bool = False) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout, qk_norm)

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
                 dropout: float = 0.0, qk_norm: bool = False) -> None:
        super().__init__()
        self.cross = CrossBlock(dim, heads, dropout, qk_norm) if cross else None
        self.self_attn = SelfBlock(dim, heads, dropout, qk_norm) if self_attention else None
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
