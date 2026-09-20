"""CLI: score an exported tokenizer on the robustness suite.

    python -m vqvae_latent_actions.ood --model runs/.../final --eval-set eval_set.npz --out reports --name n20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..data.chunks import load_eval_set
from ..models.hier_tokenizer import HierActionTokenizer
from .report import run_suite, write_report


def _floats(text: str) -> list[float]:
    return [float(v) for v in text.split(",") if v.strip()]


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="directory written by save_pretrained")
    p.add_argument("--eval-set", required=True, help="npz written by build_eval_set")
    p.add_argument("--out", required=True, help="directory for <name>.json and <name>.md")
    p.add_argument("--name", default=None, help="report name (default: the model directory's parent)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-pairs", type=int, default=256)
    p.add_argument("--sigmas", type=_floats, default=[0.0, 0.01, 0.05, 0.1, 0.25])
    p.add_argument("--factors", type=_floats, default=[0.5, 1.5, 2.0, 4.0])
    p.add_argument("--time-factors", type=_floats, default=[0.5, 2.0])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    model = HierActionTokenizer.from_pretrained(args.model).to(args.device).eval()
    eval_set = load_eval_set(args.eval_set)
    name = args.name or Path(args.model).resolve().parent.name
    result = run_suite(model, eval_set, sigmas=args.sigmas, factors=args.factors, time_factors=args.time_factors,
                       batch_size=args.batch_size, num_pairs=args.num_pairs, device=args.device, seed=args.seed)
    result["model"] = str(Path(args.model).resolve())
    result["eval_set"] = str(Path(args.eval_set).resolve())
    paths = write_report(result, args.out, name=name)
    print(f"ood report: {paths['markdown']}", flush=True)
    return result


if __name__ == "__main__":
    main()
