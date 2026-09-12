"""Reconstruction and token-length metrics over the valid (unpadded) entries, per embodiment.

Kept byte-for-byte compatible with `tokenizer_arms.metrics` so results drop straight into the arm comparison.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def masked_errors(actions: np.ndarray, recon: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    """Per-chunk errors over the valid [B,T,D] entries: mse, l1, max_abs (arrays [B])."""
    mask = np.asarray(valid, dtype=np.float64)
    diff = (np.asarray(recon, dtype=np.float64) - np.asarray(actions, dtype=np.float64))
    n = np.maximum(mask.sum(axis=(1, 2)), 1.0)
    mse = ((diff ** 2) * mask).sum(axis=(1, 2)) / n
    l1 = (np.abs(diff) * mask).sum(axis=(1, 2)) / n
    max_abs = (np.abs(diff) * mask).max(axis=(1, 2))
    return {"mse": mse, "l1": l1, "max_abs": max_abs}


class MetricAccumulator:
    def __init__(self, embodiment_ids: list[str]):
        self.embodiment_ids = list(embodiment_ids)
        self.mse, self.l1, self.max_abs, self.tokens, self.emb, self.failed = [], [], [], [], [], []

    def add(self, actions, recon, valid, embodiment_index, token_lengths, failed=None) -> None:
        e = masked_errors(actions, recon, valid)
        self.mse.append(e["mse"]); self.l1.append(e["l1"]); self.max_abs.append(e["max_abs"])
        self.tokens.append(np.asarray(token_lengths, dtype=np.int64))
        self.emb.append(np.asarray(embodiment_index, dtype=np.int64))
        self.failed.append(np.zeros(len(actions), dtype=bool) if failed is None else np.asarray(failed, dtype=bool))

    @staticmethod
    def _stats(mse, l1, max_abs, tokens, failed) -> dict:
        return {"n": int(len(mse)), "mse": float(mse.mean()), "rmse": float(np.sqrt(mse.mean())), "l1": float(l1.mean()),
                "max_abs_mean": float(max_abs.mean()), "max_abs_max": float(max_abs.max()),
                "tokens_mean": float(tokens.mean()), "tokens_std": float(tokens.std()), "tokens_p50": float(np.median(tokens)),
                "tokens_p95": float(np.percentile(tokens, 95)), "tokens_max": int(tokens.max()),
                "decode_failures": int(failed.sum())}

    def summary(self) -> dict:
        mse, l1, max_abs = np.concatenate(self.mse), np.concatenate(self.l1), np.concatenate(self.max_abs)
        tokens, emb, failed = np.concatenate(self.tokens), np.concatenate(self.emb), np.concatenate(self.failed)
        per = {}
        for i, eid in enumerate(self.embodiment_ids):
            sel = emb == i
            if sel.any():
                per[eid] = self._stats(mse[sel], l1[sel], max_abs[sel], tokens[sel], failed[sel])
        return {"total": self._stats(mse, l1, max_abs, tokens, failed), "per_embodiment": per}


def write_report(path: str | Path, arm: str, summary: dict, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"arm": arm, **(extra or {}), **summary}
    path.write_text(json.dumps(payload, indent=1))
    t = summary["total"]
    lines = [f"# {arm}", "", f"chunks={t['n']} tokens/chunk mean={t['tokens_mean']:.2f} p95={t['tokens_p95']:.0f} max={t['tokens_max']} "
             f"| rmse={t['rmse']:.5f} l1={t['l1']:.5f} max_abs(mean)={t['max_abs_mean']:.4f} decode_failures={t['decode_failures']}", "",
             "| embodiment | n | tokens | rmse | l1 | max_abs |", "|---|---|---|---|---|---|"]
    for eid, s in summary["per_embodiment"].items():
        lines.append(f"| {eid} | {s['n']} | {s['tokens_mean']:.1f} | {s['rmse']:.5f} | {s['l1']:.5f} | {s['max_abs_mean']:.4f} |")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")


__all__ = ["masked_errors", "MetricAccumulator", "write_report"]
