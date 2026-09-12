"""Comet ML tracking: online when the API is reachable, an offline archive otherwise, disabled in tests.

The API key never lives in the repo: comet_ml picks it up from COMET_API_KEY or the config file pointed to by
COMET_CONFIG (`~/.comet.config` by default). Every metric is mirrored into a local jsonl regardless of the mode.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

MODES = ("auto", "online", "offline", "disabled")


class RunLogger:
    def __init__(self, *, mode: str = "auto", project: str | None = None, workspace: str | None = None,
                 experiment_name: str | None = None, tags: list[str] | None = None,
                 offline_directory: str | Path | None = None, jsonl_path: str | Path | None = None,
                 eval_jsonl_path: str | Path | None = None, config: Mapping[str, Any] | None = None,
                 enabled: bool = True) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        # training steps and evaluations go to different files: one row per step, no mixed schemas
        self.jsonl_paths = {"train": Path(jsonl_path) if jsonl_path else None,
                            "eval": Path(eval_jsonl_path) if eval_jsonl_path else (Path(jsonl_path) if jsonl_path else None)}
        for path in {p for p in self.jsonl_paths.values() if p}:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.experiment = None
        self._mode = "disabled"
        if not enabled or mode == "disabled":
            return
        try:
            import comet_ml
        except Exception as exc:                                    # pragma: no cover - depends on the environment
            print(f"[comet] comet_ml unavailable ({type(exc).__name__}), logging to jsonl only", flush=True)
            return
        resolved = mode
        if mode == "auto":
            resolved = "online" if self._reachable() else "offline"
        kwargs = dict(project_name=project, auto_metric_logging=False, auto_param_logging=False,
                      auto_output_logging="simple", log_code=False, log_graph=False, log_env_details=True)
        try:
            if resolved == "online":
                self.experiment = comet_ml.Experiment(workspace=workspace, **kwargs)
            else:
                directory = Path(offline_directory or "comet_offline")
                directory.mkdir(parents=True, exist_ok=True)
                self.experiment = comet_ml.OfflineExperiment(offline_directory=str(directory), workspace=workspace, **kwargs)
            self._mode = resolved
        except Exception as exc:                                    # pragma: no cover - network/credentials
            print(f"[comet] could not start a {resolved} experiment ({type(exc).__name__}: {exc}); jsonl only", flush=True)
            return
        if experiment_name:
            self.experiment.set_name(experiment_name)
        if tags:
            self.experiment.add_tags(list(tags))
        if config:
            self.experiment.log_parameters(_flatten(config))
        print(f"[comet] mode={self._mode} project={project} workspace={workspace}", flush=True)

    @staticmethod
    def _reachable(url: str = "https://www.comet.com", timeout: float = 5.0) -> bool:
        try:
            import requests

            requests.head(url, timeout=timeout)
            return True
        except Exception:
            return False

    @property
    def mode(self) -> str:
        return self._mode

    def log_params(self, params: Mapping[str, Any]) -> None:
        if self.experiment is not None:
            self.experiment.log_parameters(_flatten(params))

    def log_metrics(self, metrics: Mapping[str, Any], step: int, split: str = "train") -> None:
        if self.experiment is not None:
            self.experiment.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))}, step=step)
        path = self.jsonl_paths.get(split)
        if path:
            with path.open("a") as handle:
                handle.write(json.dumps({"step": int(step), **{k: v for k, v in metrics.items()}}) + "\n")

    def log_other(self, key: str, value: Any) -> None:
        if self.experiment is not None:
            self.experiment.log_other(key, value)

    def end(self) -> None:
        if self.experiment is not None:
            self.experiment.end()


def _flatten(payload: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.update(_flatten(value, prefix=f"{name}."))
        elif isinstance(value, (list, tuple)):
            out[name] = json.dumps(list(value))
        else:
            out[name] = value
    return out


__all__ = ["RunLogger"]
