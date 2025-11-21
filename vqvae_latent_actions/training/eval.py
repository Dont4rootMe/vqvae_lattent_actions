"""Evaluation utilities."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from ..models.fsq_vqvae import FSQVQVAE
from ..utils.metrics import AverageMeter


def _move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _evaluate_single_dataloader(
    model: FSQVQVAE,
    dataloader: DataLoader,
    accelerator: Accelerator,
    max_eval_batches: int | None = None
) -> Dict[str, float]:
    """Evaluate on a single dataloader - runs on all processes in parallel."""
    # Accumulators for local process
    total_loss = 0.0
    total_recon_loss = 0.0
    total_commitment_loss = 0.0
    total_perplexity = 0.0
    total_samples = 0
    num_batches = 0

    for index, batch in enumerate(dataloader):
        if max_eval_batches is not None and index >= max_eval_batches:
            break
        
        batch = _move_batch_to_device(batch, accelerator.device)
        actions = batch['actions']
        outputs = model(actions)
        
        batch_size = actions.size(0)
        
        # Accumulate weighted sums
        total_loss += outputs.loss.detach().item() * batch_size
        total_recon_loss += outputs.recon_loss.detach().item() * batch_size
        total_commitment_loss += outputs.commitment_loss.detach().item() * batch_size
        total_perplexity += outputs.perplexity.detach().item()
        total_samples += batch_size
        num_batches += 1

    # Convert to tensors for gathering
    local_stats = torch.tensor(
        [total_loss, total_recon_loss, total_commitment_loss, total_perplexity, float(total_samples), float(num_batches)],
        device=accelerator.device
    )
    
    # Gather from all processes
    gathered_stats = accelerator.gather(local_stats)
    
    # Reshape if needed: gather returns flat tensor, reshape to [num_processes, 6]
    num_processes = accelerator.num_processes
    if gathered_stats.dim() == 1:
        gathered_stats = gathered_stats.reshape(num_processes, -1)
    
    # Compute global averages (only meaningful on main process, but computed on all for consistency)
    global_loss = gathered_stats[:, 0].sum().item()
    global_recon_loss = gathered_stats[:, 1].sum().item()
    global_commitment_loss = gathered_stats[:, 2].sum().item()
    global_perplexity = gathered_stats[:, 3].sum().item()
    global_samples = gathered_stats[:, 4].sum().item()
    global_batches = gathered_stats[:, 5].sum().item()
    
    results = {
        "loss": global_loss / global_samples if global_samples > 0 else 0.0,
        "recon_loss": global_recon_loss / global_samples if global_samples > 0 else 0.0,
        "commitment_loss": global_commitment_loss / global_samples if global_samples > 0 else 0.0,
        "perplexity": global_perplexity / global_batches if global_batches > 0 else 0.0,
    }
    
    return results


def evaluate(
    model: FSQVQVAE,
    dataloaders: Dict[str, DataLoader],
    accelerator: Accelerator,
    max_eval_batches: int | None = None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], list]:
    """
    Evaluate model on validation dataloaders.
    
    Runs on ALL processes in parallel for speed, but returns results only on main process.
    Other processes get empty results.
    """
    per_dataset: Dict[str, Dict[str, float]] = {}
    
    if not dataloaders:
        return per_dataset, {}, []

    model.eval()
    with torch.no_grad():
        # All processes participate in evaluation for speed
        for name, dataloader in dataloaders.items():
            if dataloader is None:
                continue
            per_dataset[name] = _evaluate_single_dataloader(
                model, dataloader, accelerator, max_eval_batches=max_eval_batches
            )
    model.train()

    # Compute aggregated metrics
    aggregated: Dict[str, float] = {}
    metric_names = []
    if per_dataset:
        metric_names = list(next(iter(per_dataset.values())).keys())
        for metric_name in metric_names:
            values = [metrics[metric_name] for metrics in per_dataset.values()]
            aggregated[metric_name] = sum(values) / len(values)
    
    # Return results on all processes (they're the same due to gather)
    # but only main process will log them
    return per_dataset, aggregated, metric_names


__all__ = ["evaluate"]
