"""Scoring a tokenizer on the shared eval set: reconstruction error, codebook usage, padding invariance."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..data.chunks import EvalSet
from ..models.quantizers import code_usage
from .metrics import MetricAccumulator


@torch.no_grad()
def evaluate_tokenizer(model, eval_set: EvalSet, *, batch_size: int = 1024, device: Any = None,
                       quantize: bool = True) -> dict:
    """Reconstruction metrics per embodiment plus codebook usage; token length is constant (N).

    `quantize=False` scores the continuous autoencoder, the only meaningful measurement while the quantizer
    warmup is still running; codebook usage is then reported as absent rather than as a collapsed codebook."""
    device = torch.device(device) if device is not None else next(model.parameters()).device
    was_training = model.training
    model.eval()
    accumulator = MetricAccumulator(eval_set.embodiment_ids)
    counts = torch.zeros(model.vocab_size, dtype=torch.long)
    for actions, valid, _time_valid, emb in eval_set.batches(batch_size):
        x = torch.as_tensor(actions, device=device)
        m = torch.as_tensor(valid, device=device)
        tokens, recon = model.encode_decode(x, m, quantize=quantize)
        accumulator.add(actions, recon.float().cpu().numpy(), valid, emb, [model.num_tokens] * len(actions))
        if quantize:
            counts += torch.bincount(tokens.reshape(-1).cpu(), minlength=model.vocab_size)
    summary = accumulator.summary()
    used = int((counts > 0).sum())
    probs = counts.float() / counts.sum().clamp_min(1)
    nonzero = probs[probs > 0]
    perplexity = float(torch.exp(-(nonzero * nonzero.log()).sum())) if quantize else 1.0
    summary["usage"] = {"vocab_size": model.vocab_size, "tokens_per_chunk": model.num_tokens,
                        "bits_per_chunk": model.bits_per_chunk, "quantized": bool(quantize), "codes_used": used,
                        "usage_percent": 100.0 * used / model.vocab_size, "perplexity": perplexity}
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
