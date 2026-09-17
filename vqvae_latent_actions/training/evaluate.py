"""Scoring a tokenizer on the shared eval set: reconstruction error, codebook usage, padding invariance."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from ..data.chunks import EvalSet
from ..models.quantizers import code_usage
from .metrics import MetricAccumulator


def _histogram(counts: torch.Tensor, vocab_size: int) -> dict:
    """How many codes a histogram uses and how evenly, in codes and in bits."""
    probs = counts.float() / counts.sum().clamp_min(1)
    nonzero = probs[probs > 0]
    entropy = float(-(nonzero * nonzero.log()).sum())
    used = int((counts > 0).sum())
    return {"codes_used": used, "usage_percent": 100.0 * used / vocab_size,
            "perplexity": math.exp(entropy), "bits": entropy / math.log(2)}


def _empty_histogram(vocab_size: int) -> dict:
    return {"codes_used": 0, "usage_percent": 0.0, "perplexity": 1.0, "bits": 0.0}


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
    counts = torch.zeros(model.num_tokens, model.vocab_size, dtype=torch.long)
    position_offset = torch.arange(model.num_tokens).view(1, -1) * model.vocab_size
    for actions, valid, _time_valid, emb in eval_set.batches(batch_size):
        x = torch.as_tensor(actions, device=device)
        m = torch.as_tensor(valid, device=device)
        tokens, recon = model.encode_decode(x, m, quantize=quantize)
        accumulator.add(actions, recon.float().cpu().numpy(), valid, emb, [model.num_tokens] * len(actions))
        if quantize:
            flat = tokens.cpu().long() + position_offset
            counts += torch.bincount(flat.reshape(-1), minlength=counts.numel()).view_as(counts)
    summary = accumulator.summary()
    pooled = _histogram(counts.sum(dim=0), model.vocab_size) if quantize else _empty_histogram(model.vocab_size)
    per_position = [_histogram(counts[i], model.vocab_size) for i in range(model.num_tokens)] if quantize else []
    summary["usage"] = {"vocab_size": model.vocab_size, "tokens_per_chunk": model.num_tokens,
                        "bits_per_chunk": model.bits_per_chunk, "quantized": bool(quantize), **pooled,
                        "per_position": per_position,
                        # the pooled figure hides dead positions; these two are what the bottleneck really carries
                        "min_position_perplexity": min((e["perplexity"] for e in per_position), default=1.0),
                        "effective_bits_per_chunk": float(sum(e["bits"] for e in per_position))}
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


@torch.no_grad()
def attention_logit_report(model, eval_set: EvalSet, *, num: int = 64, device: Any = None) -> dict:
    """Largest pre-softmax attention logit over every attention module, on a few chunks spread over the eval set.

    Growth here preceded the late divergence of a production run (one decoder logit reached 3e7) long before the
    loss showed anything, so it is logged with every evaluation.
    """
    from ..models.blocks import Attention
    device = torch.device(device) if device is not None else next(model.parameters()).device
    was_training = model.training
    model.eval()
    worst: dict[str, float] = {}

    def probe(name):
        def hook(module, inputs, output):
            scores = module.logits(*inputs[:3])
            finite = torch.isfinite(scores)
            if bool(finite.any()):
                worst[name] = max(worst.get(name, 0.0), float(scores[finite].abs().max()))
        return hook

    hooks = [m.register_forward_hook(probe(n)) for n, m in model.named_modules() if isinstance(m, Attention)]
    try:
        picks = np.linspace(0, len(eval_set.actions) - 1, num=min(num, len(eval_set.actions))).astype(int)
        model.encode_decode(torch.as_tensor(eval_set.actions[picks], device=device),
                            torch.as_tensor(eval_set.valid[picks], device=device))
    finally:
        for hook in hooks:
            hook.remove()
        if was_training:
            model.train()
    if not worst:                  # no finite logit anywhere: the model has diverged; report it instead of raising
        return {"max": float("nan"), "module": "non-finite"}
    module, value = max(worst.items(), key=lambda item: item[1])
    return {"max": value, "module": module}


__all__ = ["evaluate_tokenizer", "padding_invariance_mismatch", "model_report_extra", "attention_logit_report"]
