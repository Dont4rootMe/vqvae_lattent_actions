"""Metric accumulation helpers."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, List

import matplotlib.pyplot as plt


@dataclass
class AverageMeter:
    value: float = 0.0
    count: float = 0.0

    def update(self, val: float, n: float) -> None:
        self.value += val
        self.count += n

    def compute(self) -> float:
        if self.count == 0:
            return 0.0
        return self.value / self.count


class ValidationCurveTracker:
    def __init__(self) -> None:
        self.history: DefaultDict[str, List[float]] = defaultdict(list)
        self.steps: List[int] = []

    def update(self, step: int, metrics: Dict[str, float]) -> None:
        self.steps.append(step)
        for name, value in metrics.items():
            self.history[name].append(value)

    def plot(self):
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, values in self.history.items():
            ax.plot(self.steps, values, label=name)
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Metric Value")
        ax.set_title("Validation Metrics (averaged across datasets)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        return fig


__all__ = ["AverageMeter", "ValidationCurveTracker"]
