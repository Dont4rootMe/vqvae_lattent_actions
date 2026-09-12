"""Unified action-space layout: named semantic groups over the flat action vector.

The layout is data, never hardcoded in the model: it comes from the fork's yaml
(`configs/layout/bimanual_rotation6d.yaml`), from an `action_chunks` manifest, or from a dict
stored inside a model checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

ROTATION_DIMS = {"rotation_6d": 6, "quaternion": 4, "euler": 3, "axis_angle": 3, "rotation_matrix": 9}


@dataclass(frozen=True)
class UnifiedLayout:
    """Contiguous named groups tiling `[0, total_dim)` in order."""

    names: tuple[str, ...]
    starts: tuple[int, ...]
    ends: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.names or not (len(self.names) == len(self.starts) == len(self.ends)):
            raise ValueError("names, starts and ends must be non-empty and of equal length")
        position = 0
        for name, start, end in zip(self.names, self.starts, self.ends):
            if int(start) != position or int(end) <= int(start):
                raise ValueError(f"groups must tile [0, total_dim) without gaps or overlaps; {name!r} breaks at {start}")
            position = int(end)

    @property
    def total_dim(self) -> int:
        return int(self.ends[-1])

    @property
    def num_groups(self) -> int:
        return len(self.names)

    def intervals(self) -> dict[str, tuple[int, int]]:
        return {n: (int(s), int(e)) for n, s, e in zip(self.names, self.starts, self.ends)}

    def group_index(self) -> torch.Tensor:
        """[D] long: group id of every dimension."""
        out = torch.zeros(self.total_dim, dtype=torch.long)
        for g, (s, e) in enumerate(zip(self.starts, self.ends)):
            out[int(s):int(e)] = g
        return out

    def membership(self) -> torch.Tensor:
        """[G, D] bool: True where the dimension belongs to the group."""
        out = torch.zeros(self.num_groups, self.total_dim, dtype=torch.bool)
        for g, (s, e) in enumerate(zip(self.starts, self.ends)):
            out[g, int(s):int(e)] = True
        return out

    def dim_weights(self, group_weights: Mapping[str, float] | None) -> torch.Tensor:
        """[D] float: per-dimension loss weight, 1.0 unless the group is listed."""
        out = torch.ones(self.total_dim, dtype=torch.float32)
        for name, weight in (group_weights or {}).items():
            if name not in self.names:
                raise KeyError(f"unknown group {name!r}; known: {list(self.names)}")
            s, e = self.intervals()[name]
            out[s:e] = float(weight)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"names": list(self.names), "starts": [int(s) for s in self.starts], "ends": [int(e) for e in self.ends]}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UnifiedLayout":
        return cls(tuple(payload["names"]), tuple(int(s) for s in payload["starts"]), tuple(int(e) for e in payload["ends"]))

    @classmethod
    def from_intervals(cls, intervals: Mapping[str, tuple[int, int]]) -> "UnifiedLayout":
        items = sorted(intervals.items(), key=lambda kv: kv[1][0])
        return cls(tuple(k for k, _ in items), tuple(int(v[0]) for _, v in items), tuple(int(v[1]) for _, v in items))

    @classmethod
    def from_manifest(cls, manifest: Any) -> "UnifiedLayout":
        """`manifest` is an action_chunks Manifest (or anything with `layout_intervals()`)."""
        return cls.from_intervals(manifest.layout_intervals())

    @classmethod
    def from_yaml(cls, path: str | Path, rotation_representation: str | None = None) -> "UnifiedLayout":
        payload = yaml.safe_load(Path(path).read_text())
        rotation = rotation_representation or payload.get("rotation_representation", "rotation_6d")
        if rotation not in ROTATION_DIMS:
            raise ValueError(f"unknown rotation representation {rotation!r}")
        names, starts, ends, position = [], [], [], 0
        for name, group in payload["groups"].items():
            kind = group.get("kind", "vector")
            if kind == "rotation":
                dim = ROTATION_DIMS[rotation]
            elif kind == "vector":
                dim = int(group["dim"] if "dim" in group else group["max_dim"])
            else:
                raise ValueError(f"unknown group kind {kind!r} for {name!r}")
            names.append(name); starts.append(position); ends.append(position + dim); position += dim
        return cls(tuple(names), tuple(starts), tuple(ends))


__all__ = ["UnifiedLayout", "ROTATION_DIMS"]
