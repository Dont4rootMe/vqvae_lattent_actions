"""Run the whole robustness suite over one model and write the numbers as json plus a readable table."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..data.chunks import EvalSet
from ..layout import UnifiedLayout
from . import robustness


def run_suite(model, eval_set: EvalSet, *, layout: UnifiedLayout | None = None, sigmas=(0.0, 0.01, 0.05, 0.1, 0.25),
              factors=(0.5, 1.5, 2.0, 4.0), time_factors=(0.5, 2.0), batch_size: int = 512, num_pairs: int = 256,
              device: Any = None, seed: int = 0) -> dict:
    layout = layout or UnifiedLayout.from_dict(model.config.layout)
    _, recon, clean = robustness._clean_pass(model, eval_set, batch_size=batch_size, device=device)
    common = {"batch_size": batch_size, "device": device}
    return {
        "chunks": len(eval_set),
        "clean": clean,
        "noise": robustness.noise_sweep(model, eval_set, sigmas=sigmas, seed=seed, **common),
        "amplitude": robustness.amplitude_sweep(model, eval_set, factors=factors, **common),
        "time": robustness.time_sweep(model, eval_set, factors=time_factors, **common),
        "groups": robustness.group_dropout(model, eval_set, layout, **common),
        "interpolation": robustness.latent_interpolation(model, eval_set, num_pairs=num_pairs, seed=seed,
                                                         device=device),
    }


def _table(rows: list[dict], key: str, title: str) -> list[str]:
    if not rows:
        return []
    out = [f"## {title}", "",
           f"| {title.lower()} | rmse | l1 | tokens identical | token agreement | input shift | output shift |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r[key]} | {r['rmse']:.5f} | {r['l1']:.5f} | {r['tokens_identical']:.3f} | "
                   f"{r['token_agreement']:.3f} | {r['input_shift']:.4f} | {r['output_shift']:.4f} |")
    return out + [""]


def render_markdown(result: dict, name: str) -> str:
    clean = result["clean"]
    lines = [f"# {name}", "",
             f"chunks={result['chunks']} | clean rmse={clean['rmse']:.5f} l1={clean['l1']:.5f}", ""]
    lines += _table(result["noise"], "sigma", "Noise")
    lines += _table(result["amplitude"], "factor", "Amplitude")
    lines += _table(result["time"], "factor", "Time")
    lines += _table(result["groups"], "group", "Group dropped")
    interp = result["interpolation"]
    lines += ["## Latent interpolation", "",
              f"pairs={interp['num_pairs']} midpoint_ratio={interp['midpoint_ratio']:.3f} "
              f"monotone_fraction={interp['monotone_fraction']:.3f} "
              f"endpoint_distance={interp['endpoint_distance']:.4f}", ""]
    return "\n".join(lines)


def write_report(result: dict, directory: str | Path, *, name: str) -> dict[str, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    json_path, md_path = directory / f"{name}.json", directory / f"{name}.md"
    json_path.write_text(json.dumps(result, indent=2))
    md_path.write_text(render_markdown(result, name))
    return {"json": json_path, "markdown": md_path}


__all__ = ["render_markdown", "run_suite", "write_report"]
