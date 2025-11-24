"""GMB-based VQ-VAE model for discrete action tokenization."""
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
class GMBVQVAEOutput:
    loss: Tensor
    recon_loss: Tensor
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


class GMBQuantizer(nn.Module):
    """VQ-VAE quantizer using Gumbel-Softmax trick for discrete latent representation."""

    def __init__(self, gamma: float, latent_dim: int, seq_len: int, quant_layers: int, vocab_size: int) -> None:
        super().__init__()
        self.gamma = gamma
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.quant_layers = quant_layers
        self.vocab_size = vocab_size
        self.embedding_dim = seq_len * (latent_dim // (quant_layers * 3))
        self.quantized_dim = quant_layers * seq_len * (latent_dim // (quant_layers * 3))
        
        # Small network to compress sequence to logits
        self.to_logits = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, quant_layers * vocab_size)
        )
        
        # Shared embedding layer for all quantization layers
        self.embeddings = nn.Embedding(vocab_size, self.embedding_dim)
        
        # Network to expand codes back to full sequence representation
        # Input: latent_dim (quant_layers * embedding_dim), Output: seq_len * latent_dim
        self.from_codes = nn.Sequential(
            nn.Linear(self.quantized_dim, self.quantized_dim * 2),
            nn.GELU(),
            nn.Linear(self.quantized_dim * 2, seq_len * latent_dim)
        )

    def forward(
        self, 
        latents: Tensor, 
        gamma: Optional[float] = None, 
        deterministic: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """
        Quantize latents using Gumbel-Softmax trick.
        
        Args:
            latents: Tensor of shape (B, seq_len, latent_dim)
            gamma: Optional temperature override for Gumbel-Softmax
            deterministic: If True, use argmax instead of Gumbel-Softmax sampling
        
        Returns:
            indices: Tensor of shape (B, quant_layers) containing discrete codes
            reconstructed: Tensor of shape (B, seq_len, latent_dim) reconstructed latents
        """
        temperature = gamma if gamma is not None else self.gamma
        batch_size, seq_len, _ = latents.shape
        
        object_latent = latents[:, 0, :]
        object_logits = self.to_logits(object_latent) # [B x quant_layers x vocab_size]
        object_logits = object_logits.view(batch_size, self.quant_layers, self.vocab_size) # [B x quant_layers x vocab_size]
        
        if deterministic:
            # Deterministic: use argmax
            indices = object_logits.argmax(dim=-1)  # [B x quant_layers]
        else:
            # Stochastic: apply Gumbel-Softmax
            soft_codes = F.gumbel_softmax(
                object_logits, 
                tau=temperature, 
                hard=True, 
                dim=-1
            )  # [B x quant_layers x vocab_size]
            indices = soft_codes.argmax(dim=-1)  # [B x quant_layers]
        
        quantized = self.embeddings(indices)  # [B x quant_layers x (seq_len * (latent_dim // (quant_layers * 3)))]
        quantized_flat = quantized.reshape(batch_size, -1)  # [B x seq_len * (latent_dim // 3)]
        reconstructed_flat = self.from_codes(quantized_flat)  # [B x seq_len * latent_dim]
        reconstructed = reconstructed_flat.view(batch_size, seq_len, self.latent_dim)

        return reconstructed, indices



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


class GMBVQVAE(nn.Module):
    """GMB-VQ-VAE for sequences of robotic actions."""

    def __init__(
        self,
        action_dim: int,
        latent_dim: int,
        hidden_dim: int,
        num_heads: int,
        action_seq_len: int,
        gmb_gamma: float,
        quant_layers: int,
        vocab_size: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        commitment_cost: float = 0.25,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
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
        self.quantizer = GMBQuantizer(
            gamma=gmb_gamma, 
            latent_dim=latent_dim,
            seq_len=action_seq_len,
            quant_layers=quant_layers,
            vocab_size=vocab_size,
        )
        
        self.config = config or self._build_config(
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            action_seq_len=action_seq_len,
            gmb_gamma=gmb_gamma,
            quant_layers=quant_layers,
            vocab_size=vocab_size,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            commitment_cost=commitment_cost,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

    def _build_config(self, **kwargs: Any) -> Dict[str, Any]:
        return kwargs

    def encode(self, actions: Tensor) -> Tensor:
        return self.encoder(actions)

    def decode(self, latents: Tensor) -> Tensor:
        return self.decoder(latents)

    def quantize(self, latents: Tensor) -> Tuple[Tensor, Tensor]:
        quantized, indices = self.quantizer(latents)
        return quantized, indices

    def forward(self, actions: Tensor) -> GMBVQVAEOutput:  # type: ignore[override]
        latents = self.encode(actions)
        quantized, indices = self.quantize(latents)
        reconstructions = self.decode(quantized)
        recon_loss = F.mse_loss(reconstructions, actions)
        
        # Note: The vector_quantize_pytorch FSQ implementation might already handle 
        # straight-through gradients, but we should check if we need commitment loss.
        # Usually FSQ relies on the quantizer's internal mechanics or implicit commitment via STE.
        # However, adding an explicit auxiliary loss to pull encoder outputs to valid states is good.
        # The library implementation returns quantized values with STE.
        # We will keep our manual commitment loss calculation as it helps convergence.
        
        commitment_loss = self.commitment_cost * F.mse_loss(latents, quantized.detach())
        
        loss = recon_loss + commitment_loss
        
        return GMBVQVAEOutput(
            loss=loss,
            recon_loss=recon_loss,
            reconstructions=reconstructions,
            latents=latents,
            quantized_latents=quantized,
            indices=indices,
        )

    def compute_loss(self, batch: Dict[str, Tensor]) -> GMBVQVAEOutput:
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
    ) -> "GMBVQVAE":
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


__all__ = ["GMBVQVAE", "GMBVQVAEOutput", "GMBQuantizer"]
