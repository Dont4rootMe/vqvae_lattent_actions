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
    def forward(self, actions: Tensor, mask: Tensor, quantize: bool = True) -> dict[str, Tensor]:
        """`quantize=False` trains the plain autoencoder: useful as a warmup so the latents become informative
        before the grid is imposed on them."""
        latents = self.encode_continuous(actions, mask)
        quantized = self.quantize(latents)
        codes = quantized.codes if quantize else latents
        recon = self.decode_latents(codes, mask)
        target = actions.masked_fill(~mask, 0.0).to(recon.dtype)
        weights = mask.to(recon.dtype) * self.dim_weights.view(1, 1, -1).to(recon.dtype)
        recon_mse = (((recon - target) ** 2) * weights).sum() / weights.sum().clamp_min(1.0)
        loss = recon_mse + (quantized.aux_loss.to(recon_mse.dtype) if quantize else recon_mse.new_zeros(()))
        reported_aux = quantized.aux_loss.detach() if quantize else recon_mse.new_zeros(())
        return {"loss": loss, "recon_mse": recon_mse.detach(), "aux_loss": reported_aux,
                "indices": quantized.indices, "recon": recon, "latents": latents}

    @torch.no_grad()
    def tokenize(self, actions: Tensor, mask: Tensor) -> Tensor:
        return self.quantize(self.encode_continuous(actions, mask)).indices

    @torch.no_grad()
    def detokenize(self, tokens: Tensor, mask: Tensor) -> Tensor:
        return self.decode_latents(self.quantizer.indices_to_codes(tokens), mask)

    @torch.no_grad()
    def encode_decode(self, actions: Tensor, mask: Tensor, quantize: bool = True) -> tuple[Tensor, Tensor]:
        """Tokens are always returned; `quantize=False` reconstructs from the continuous latents instead of the
        grid, which is what the model is actually trained on during the quantizer warmup."""
        latents = self.encode_continuous(actions, mask)
        quantized = self.quantize(latents)
        return quantized.indices, self.decode_latents(quantized.codes if quantize else latents, mask)

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
