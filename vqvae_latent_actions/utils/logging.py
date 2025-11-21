"""Logging helpers (console + ClearML)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from accelerate import Accelerator

try:  # pragma: no cover - optional dependency
    from clearml import Task
except Exception:  # pragma: no cover - clearml not installed
    Task = None  # type: ignore[assignment]


@dataclass
class ClearMLConfig:
    project_name: str
    task_name: str
    tags: Optional[list[str]] = None
    enable: bool = True


class ClearMLLogger:
    """Thin wrapper around ClearML's logger."""

    def __init__(self, config: ClearMLConfig) -> None:
        if Task is None:
            raise ImportError("clearml is not installed but ClearML logging is enabled")
        self.task = Task.init(
            project_name=config.project_name,
            task_name=config.task_name,
            tags=config.tags,
            reuse_last_task_id=False,
        )
        self.logger = self.task.get_logger()

    def log_metrics(self, split: str, metrics: Dict[str, float], step: int) -> None:
        for name, value in metrics.items():
            self.logger.report_scalar(title=split, series=name, value=value, iteration=step)

    def log_figure(self, title: str, series: str, figure, step: int) -> None:
        self.logger.report_matplotlib_figure(title=title, series=series, figure=figure, iteration=step)


class MetricLogger:
    """Routes metrics to the console and optional ClearML logger."""

    def __init__(self, accelerator: Accelerator, clearml_logger: Optional[ClearMLLogger] = None) -> None:
        self.accelerator = accelerator
        self.clearml_logger = clearml_logger

    def log(self, split: str, metrics: Dict[str, float], step: int) -> None:
        if self.accelerator.is_main_process:
            msg = f"[{split}] step={step} " + " ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            self.accelerator.print(msg)
            if self.clearml_logger is not None:
                self.clearml_logger.log_metrics(split, metrics, step)

    def log_figure(self, title: str, series: str, figure, step: int) -> None:
        if self.accelerator.is_main_process and self.clearml_logger is not None:
            self.clearml_logger.log_figure(title, series, figure, step)


def maybe_create_clearml_logger(cfg: Dict, accelerator: Accelerator) -> Optional[ClearMLLogger]:
    clearml_cfg = cfg.get("clearml")
    if not clearml_cfg or not clearml_cfg.get("enable", True):
        return None
    if not accelerator.is_main_process:
        return None
    config = ClearMLConfig(
        project_name=clearml_cfg.get("project_name", "vqvae"),
        task_name=clearml_cfg.get("task_name", "training"),
        tags=clearml_cfg.get("tags"),
        enable=True,
    )
    return ClearMLLogger(config)


__all__ = ["MetricLogger", "maybe_create_clearml_logger", "ClearMLLogger", "ClearMLConfig"]
