"""FSQ-based VQ-VAE model for discrete action tokenization."""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class FSQVQVAEOutput:
    loss: Tensor
    recon_loss: Tensor
    commitment_loss: Tensor
    perplexity: Tensor
    reconstructions: Tensor
    latents: Tensor
    quantized_latents: Tensor
    indices: Tensor


class SinusoidalPositionalEncoding(nn.Module):
    """Adds sinusoidal positional encodings to a sequence."""

    def __init__(self, dim: int, max_len: int = 2048) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return x + self.pe[:, : x.size(1)]


class FSQQuantizer(nn.Module):
    """Finite Scalar Quantizer that deterministically maps latents to discrete bins."""

    def __init__(self, levels: Sequence[int], clip_range: float = 1.0) -> None:
        super().__init__()
        if not levels:
            raise ValueError("levels must be a non-empty sequence")
        if any(l <= 1 for l in levels):
            raise ValueError("each entry in levels must be > 1")
        self.levels = list(int(l) for l in levels)
        self.clip_range = float(clip_range)
        self.group_dim = len(self.levels)
        self.codebook_size = 1
        for l in self.levels:
            self.codebook_size *= l

    def extra_repr(self) -> str:
        return f"levels={self.levels}, clip_range={self.clip_range}"

    def _build_grid(self, level: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.linspace(-self.clip_range, self.clip_range, steps=level, device=device, dtype=dtype)

    def quantize(self, chunk: Tensor) -> Tuple[Tensor, Tensor]:
        """Quantize a latent chunk of shape (N, group_dim)."""

        if chunk.dim() != 2 or chunk.size(-1) != self.group_dim:
            raise ValueError(
                f"chunk must have shape (N, {self.group_dim}), got {tuple(chunk.shape)}"
            )

        device = chunk.device
        dtype = chunk.dtype
        chunk = chunk.clamp(-self.clip_range, self.clip_range)
        quantized = torch.empty_like(chunk)
        flat_indices = torch.zeros(chunk.size(0), device=device, dtype=torch.long)
        multiplier = 1
        for dim_idx, level in enumerate(self.levels):
            grid = self._build_grid(level, device=device, dtype=dtype)
            values = chunk[:, dim_idx].unsqueeze(1)
            distances = torch.abs(values - grid.unsqueeze(0))
            best_idx = torch.argmin(distances, dim=1)
            quantized[:, dim_idx] = grid[best_idx]
            flat_indices += best_idx * multiplier
            multiplier *= level
        return quantized, flat_indices

    def forward(self, latents: Tensor) -> Tuple[Tensor, Tensor]:  # type: ignore[override]
        """Quantize latents of shape (B, T, D)."""

        if latents.dim() != 3:
            raise ValueError("latents must have shape (batch, seq, latent_dim)")

        bsz, seq_len, dim = latents.shape
        if dim % self.group_dim != 0:
            raise ValueError(
                f"latent_dim ({dim}) must be divisible by the FSQ group dimension ({self.group_dim})"
            )

        num_groups = dim // self.group_dim
        latents_chunks = latents.reshape(-1, num_groups, self.group_dim)
        latents_chunks = latents_chunks.reshape(-1, self.group_dim)
        quantized_chunks, indices = self.quantize(latents_chunks)
        quantized = quantized_chunks.reshape(bsz, seq_len, num_groups * self.group_dim)
        indices = indices.reshape(bsz, seq_len, num_groups)
        return quantized, indices

    def perplexity(self, indices: Tensor) -> Tensor:
        """
        Compute perplexity averaged across all groups.
        
        Args:
            indices: Tensor of shape (batch, seq, num_groups) containing discrete codes
        
        Returns:
            Scalar tensor with average perplexity across all groups
        """
        with torch.no_grad():
            if indices.dim() == 3:
                # Multi-group case: compute perplexity for each group separately
                batch_size, seq_len, num_groups = indices.shape
                total_perplexity = 0.0
                
                for group_idx in range(num_groups):
                    group_indices = indices[:, :, group_idx].reshape(-1)
                    hist = torch.bincount(group_indices, minlength=self.codebook_size).float()
                    probs = hist / (hist.sum() + 1e-8)
                    non_zero = probs[probs > 0]
                    entropy = -(non_zero * torch.log(non_zero)).sum()
                    total_perplexity += torch.exp(entropy).item()
                
                return torch.tensor(total_perplexity / num_groups, device=indices.device)
            else:
                # Single group case: compute perplexity directly
                hist = torch.bincount(indices.view(-1), minlength=self.codebook_size).float()
                probs = hist / (hist.sum() + 1e-8)
                non_zero = probs[probs > 0]
                entropy = -(non_zero * torch.log(non_zero)).sum()
                return torch.exp(entropy)


class TransformerBackbone(nn.Module):
    """Lightweight Transformer encoder used for both encoder and decoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.positional_encoding = SinusoidalPositionalEncoding(hidden_dim, max_len=max_seq_len)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = self.input_proj(x)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        x = self.layer_norm(x)
        return self.output_proj(x)


class FSQVQVAE(nn.Module):
    """FSQ-VQ-VAE for sequences of robotic actions."""

    def __init__(
        self,
        action_dim: int,
        latent_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        fsq_levels: Sequence[int],
        commitment_cost: float = 0.25,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        clip_range: float = 1.0,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if latent_dim % len(fsq_levels) != 0:
            raise ValueError("latent_dim must be divisible by the number of FSQ levels")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.commitment_cost = commitment_cost

        self.encoder = TransformerBackbone(
            input_dim=action_dim,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        self.decoder = TransformerBackbone(
            input_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        self.quantizer = FSQQuantizer(levels=fsq_levels, clip_range=clip_range)
        self.config = config or self._build_config(
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            fsq_levels=list(fsq_levels),
            commitment_cost=commitment_cost,
            dropout=dropout,
            max_seq_len=max_seq_len,
            clip_range=clip_range,
        )

    def _build_config(self, **kwargs: Any) -> Dict[str, Any]:
        return kwargs

    def encode(self, actions: Tensor) -> Tensor:
        return self.encoder(actions)

    def decode(self, latents: Tensor) -> Tensor:
        return self.decoder(latents)

    def quantize(self, latents: Tensor) -> Tuple[Tensor, Tensor]:
        quantized, indices = self.quantizer(latents)
        quantized = latents + (quantized - latents).detach()
        return quantized, indices

    def forward(self, actions: Tensor) -> FSQVQVAEOutput:  # type: ignore[override]
        latents = self.encode(actions)
        quantized, indices = self.quantize(latents)
        reconstructions = self.decode(quantized)
        recon_loss = F.mse_loss(reconstructions, actions)
        # FSQ commitment loss: encourage encoder to match quantized values
        # Note: detach on quantized, NOT on latents (unlike VQ-VAE)
        commitment_loss = self.commitment_cost * F.mse_loss(latents, quantized.detach())
        loss = recon_loss + commitment_loss
        perplexity = self.quantizer.perplexity(indices)
        return FSQVQVAEOutput(
            loss=loss,
            recon_loss=recon_loss,
            commitment_loss=commitment_loss,
            perplexity=perplexity,
            reconstructions=reconstructions,
            latents=latents,
            quantized_latents=quantized,
            indices=indices,
        )

    def compute_loss(self, batch: Dict[str, Tensor]) -> FSQVQVAEOutput:
        actions = batch["actions"].float()
        return self.forward(actions)

    def save_pretrained(self, save_directory: str, filename: str = "pytorch_model.bin") -> None:
        os.makedirs(save_directory, exist_ok=True)
        model_path = os.path.join(save_directory, filename)
        config_path = os.path.join(save_directory, "config.json")
        torch.save(self.state_dict(), model_path)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)

    def save_state(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def from_pretrained(
        cls,
        directory: str,
        map_location: Optional[str | torch.device] = None,
        strict: bool = True,
    ) -> "FSQVQVAE":
        config_path = os.path.join(directory, "config.json")
        model_path = os.path.join(directory, "pytorch_model.bin")
        if not os.path.exists(config_path) or not os.path.exists(model_path):
            raise FileNotFoundError(f"Expected config and model files under {directory}")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        model = cls(**config)
        state_dict = torch.load(model_path, map_location=map_location)
        model.load_state_dict(state_dict, strict=strict)
        return model

    def to_config(self) -> Dict[str, Any]:
        return dict(self.config)


__all__ = ["FSQVQVAE", "FSQVQVAEOutput", "FSQQuantizer"]
