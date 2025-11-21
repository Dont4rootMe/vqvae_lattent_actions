"""Evaluation utilities."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from ..models.fsq_vqvae import FSQVQVAE
from ..utils.metrics import AverageMeter


def _move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _gather_sum(accelerator: Accelerator, tensor: torch.Tensor) -> float:
    gathered = accelerator.gather_for_metrics(tensor)
    return gathered.sum().item()


def _evaluate_single_dataloader(
    model: FSQVQVAE,
    dataloader: DataLoader,
    accelerator: Accelerator,
) -> Dict[str, float]:
    metrics = {
        "loss": AverageMeter(),
        "recon_loss": AverageMeter(),
        "commitment_loss": AverageMeter(),
    }
    perplexity_meter = AverageMeter()

    for batch in dataloader:
        batch = _move_batch_to_device(batch, accelerator.device)
        outputs = model.compute_loss(batch)
        batch_size = torch.tensor(batch["actions"].size(0), device=accelerator.device, dtype=torch.float32)
        weight = accelerator.gather_for_metrics(batch_size)
        total_weight = weight.sum().item()

        loss = outputs.loss.detach() * batch_size
        recon_loss = outputs.recon_loss.detach() * batch_size
        commitment_loss = outputs.commitment_loss.detach() * batch_size

        metrics["loss"].update(_gather_sum(accelerator, loss), total_weight)
        metrics["recon_loss"].update(_gather_sum(accelerator, recon_loss), total_weight)
        metrics["commitment_loss"].update(
            _gather_sum(accelerator, commitment_loss), total_weight
        )

        perp = outputs.perplexity.detach()
        perp_sum = _gather_sum(accelerator, perp)
        perplexity_meter.update(perp_sum, float(accelerator.num_processes))

    results = {name: meter.compute() for name, meter in metrics.items()}
    results["perplexity"] = perplexity_meter.compute()
    return results


def evaluate(
    model: FSQVQVAE,
    dataloaders: Dict[str, DataLoader],
    accelerator: Accelerator,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    per_dataset: Dict[str, Dict[str, float]] = {}
    if not dataloaders:
        return per_dataset, {}

    model.eval()
    with torch.no_grad():
        for name, dataloader in dataloaders.items():
            if dataloader is None:
                continue
            per_dataset[name] = _evaluate_single_dataloader(model, dataloader, accelerator)
    model.train()

    aggregated: Dict[str, float] = {}
    if per_dataset:
        metric_names = next(iter(per_dataset.values())).keys()
        for metric_name in metric_names:
            values = [metrics[metric_name] for metrics in per_dataset.values()]
            aggregated[metric_name] = sum(values) / len(values)
    return per_dataset, aggregated


__all__ = ["evaluate"]
