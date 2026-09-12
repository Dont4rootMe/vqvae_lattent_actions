"""Chunks of the R0_v2.1 mixture as the VLA sees them: actions [T, 119] plus a {0,1} mask of real entries.

Training samples with the VLA mixture weights (WeightedChunkDataset); the shared eval set is a deterministic,
evenly spread pass over held-out episodes, stored as npz so every tokenizer is scored on the very same chunks.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from action_chunks.datasets import SequentialChunkDataset, WeightedChunkDataset, collate_chunks
from action_chunks.manifest import Manifest

from ..layout import UnifiedLayout

HOLDOUT_EVERY = 50   # embodiments without a validation list hold out every 50th episode (2%)


def load_manifest(manifest: str | Path | Manifest) -> Manifest:
    return manifest if isinstance(manifest, Manifest) else Manifest.load(manifest)


def layout_from_manifest(manifest: str | Path | Manifest) -> UnifiedLayout:
    return UnifiedLayout.from_manifest(load_manifest(manifest))


def batch_to_inputs(batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(actions, mask, embodiment_index). Mask is True only where the value is a real action."""
    actions = batch["fast_action"].float()
    time_valid = ~batch["fast_action_is_pad"].bool()
    mask = ~batch["fast_action_pad_mask"].bool() & time_valid.unsqueeze(-1)
    actions = actions.masked_fill(~mask, 0.0)
    return actions, mask, batch["embodiment_index"].long()


def train_loader(manifest: str | Path, cache_dir: str | Path | None, *, batch_size: int, num_workers: int,
                 seed: int = 0, samples_per_episode: int = 8, epoch_length: int | None = None, split: str = "train",
                 holdout_every: int | None = HOLDOUT_EVERY, pin_memory: bool = False) -> DataLoader:
    dataset = WeightedChunkDataset(load_manifest(manifest), split=split, cache_dir=cache_dir, seed=seed,
                                   samples_per_episode=samples_per_episode, epoch_length=epoch_length,
                                   holdout_every=holdout_every)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_chunks,
                      drop_last=True, pin_memory=pin_memory, persistent_workers=num_workers > 0)


def sequential_loader(manifest: str | Path, cache_dir: str | Path | None, *, split: str = "val", stride: int = 1,
                      max_chunks_per_embodiment: int | None = None, batch_size: int = 256, num_workers: int = 8,
                      embodiments: list[str] | None = None, holdout_every: int | None = HOLDOUT_EVERY) -> DataLoader:
    dataset = SequentialChunkDataset(load_manifest(manifest), stride=stride, split=split, cache_dir=cache_dir,
                                     embodiments=embodiments, max_chunks_per_embodiment=max_chunks_per_embodiment,
                                     holdout_every=holdout_every)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_chunks)


@dataclass
class EvalSet:
    actions: np.ndarray            # [N, T, D] float32
    valid: np.ndarray              # [N, T, D] bool
    time_valid: np.ndarray         # [N, T] bool
    embodiment_index: np.ndarray   # [N] int64
    embodiment_ids: list[str]
    manifest_sha: str

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    def batches(self, batch_size: int):
        for start in range(0, len(self), batch_size):
            sl = slice(start, start + batch_size)
            yield self.actions[sl], self.valid[sl], self.time_valid[sl], self.embodiment_index[sl]


def build_eval_set(manifest: str | Path, cache_dir: str | Path | None, out_path: str | Path, *,
                   per_embodiment: int = 500, split: str = "val", num_workers: int = 8, batch_size: int = 256,
                   seed: int = 0, holdout_every: int | None = HOLDOUT_EVERY) -> EvalSet:
    """At most `per_embodiment` evenly spread chunks from the held-out episodes of every embodiment."""
    manifest = load_manifest(manifest)
    rng = np.random.default_rng(seed)
    parts: dict[str, list[np.ndarray]] = {k: [] for k in ("actions", "valid", "time_valid", "emb")}
    for spec in manifest.embodiments:
        loader = sequential_loader(manifest, cache_dir, split=split, stride=1, max_chunks_per_embodiment=per_embodiment,
                                   batch_size=batch_size, num_workers=num_workers, embodiments=[spec.embodiment_id],
                                   holdout_every=holdout_every)
        got: dict[str, list[np.ndarray]] = {k: [] for k in parts}
        for batch in loader:
            actions, mask, emb = batch_to_inputs(batch)
            got["actions"].append(actions.numpy())
            got["valid"].append(mask.numpy())
            got["time_valid"].append((~batch["fast_action_is_pad"].bool()).numpy())
            got["emb"].append(emb.numpy())
        if not got["actions"]:
            print(f"eval set: {spec.embodiment_id:<48} {0:>6} chunks (no {split} episodes)", flush=True)
            continue
        merged = {k: np.concatenate(v) for k, v in got.items()}
        found = len(merged["actions"])
        if found > per_embodiment:
            keep = np.sort(rng.choice(found, size=per_embodiment, replace=False))
            merged = {k: v[keep] for k, v in merged.items()}
        for key in parts:
            parts[key].append(merged[key])
        print(f"eval set: {spec.embodiment_id:<48} {len(merged['actions']):>6} chunks (pass yielded {found})", flush=True)
    eval_set = EvalSet(actions=np.concatenate(parts["actions"]), valid=np.concatenate(parts["valid"]),
                       time_valid=np.concatenate(parts["time_valid"]), embodiment_index=np.concatenate(parts["emb"]),
                       embodiment_ids=[e.embodiment_id for e in manifest.embodiments], manifest_sha=manifest.sha256)
    save_eval_set(eval_set, out_path)
    return eval_set


def save_eval_set(eval_set: EvalSet, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, actions=eval_set.actions, valid=eval_set.valid, time_valid=eval_set.time_valid,
                        embodiment_index=eval_set.embodiment_index,
                        embodiment_ids=np.array(json.dumps(eval_set.embodiment_ids)),
                        manifest_sha=np.array(eval_set.manifest_sha))


def load_eval_set(path: str | Path) -> EvalSet:
    z = np.load(path)
    return EvalSet(actions=z["actions"].astype(np.float32), valid=z["valid"].astype(bool),
                   time_valid=z["time_valid"].astype(bool), embodiment_index=z["embodiment_index"].astype(np.int64),
                   embodiment_ids=json.loads(str(z["embodiment_ids"])), manifest_sha=str(z["manifest_sha"]))


__all__ = ["EvalSet", "HOLDOUT_EVERY", "batch_to_inputs", "build_eval_set", "layout_from_manifest", "load_eval_set",
           "load_manifest", "save_eval_set", "sequential_loader", "train_loader"]
