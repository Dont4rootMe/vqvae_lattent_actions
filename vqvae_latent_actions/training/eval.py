"""Evaluation utilities."""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from ..models.fsq_vqvae import FSQVQVAE
from ..utils.metrics import AverageMeter


def _move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _get_model_dtype(model: torch.nn.Module) -> torch.dtype:
    param = next(model.parameters(), None)
    if param is not None:
        return param.dtype
    buffer = next(model.buffers(), None)
    if buffer is not None:
        return buffer.dtype
    return torch.float32


def _ensure_actions_dtype(batch: Dict, model: torch.nn.Module) -> Dict:
    actions = batch.get("actions")
    if not isinstance(actions, torch.Tensor):
        return batch
    target_dtype = _get_model_dtype(model)
    if actions.dtype != target_dtype:
        batch["actions"] = actions.to(dtype=target_dtype)
    return batch


def _evaluate_single_dataloader(
    model: torch.nn.Module,
    dataloader: DataLoader,
    accelerator: Accelerator,
    max_eval_batches: int | None = None,
) -> Tuple[Dict[str, float], torch.Tensor | None]:
    """Evaluate on a single dataloader - runs on all processes in parallel."""
    # Accumulators for local process
    totals: Dict[str, float] = {}
    total_samples = 0.0
    accumulated_token_counts = None

    for index, batch in enumerate(dataloader):
        if max_eval_batches is not None and index >= max_eval_batches:
            break
        
        batch = _move_batch_to_device(batch, accelerator.device)
        batch = _ensure_actions_dtype(batch, model)
        actions = batch['actions']
        outputs = model(actions)
        
        batch_size = actions.size(0)
        total_samples += batch_size
        
        # Dynamically extract metrics
        if hasattr(outputs, '__dataclass_fields__'):
            field_names = outputs.__dataclass_fields__.keys()
        else:
            field_names = [name for name in dir(outputs) if not name.startswith("_")]

        for name in field_names:
            value = getattr(outputs, name)
            
            # Special handling for token_counts - accumulate histogram
            if name == 'token_counts' and isinstance(value, torch.Tensor):
                if accumulated_token_counts is None:
                    accumulated_token_counts = value.detach().clone()
                else:
                    accumulated_token_counts += value.detach()
            # Only accumulate scalar tensors for regular metrics
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                val = value.detach().item()
                if name not in totals:
                    totals[name] = 0.0
                totals[name] += val * batch_size

    # Return empty if no samples or no metrics
    if total_samples == 0 or not totals:
        return {}, None

    # Convert to tensors for gathering
    # Sort keys to ensure consistent order across processes
    metric_names = sorted(totals.keys())
    
    # Construct tensor: [total_samples, val1, val2, ...]
    local_stats = [total_samples] + [totals[name] for name in metric_names]
    local_tensor = torch.tensor(local_stats, device=accelerator.device)
    
    # Gather from all processes
    gathered_stats = accelerator.gather(local_tensor)
    
    # Reshape if needed
    num_processes = accelerator.num_processes
    if gathered_stats.dim() == 1:
        gathered_stats = gathered_stats.reshape(num_processes, -1)
    
    # Compute global sums
    global_sums = gathered_stats.sum(dim=0)
    global_samples = global_sums[0].item()
    
    results = {}
    if global_samples > 0:
        for i, name in enumerate(metric_names):
            # Index offset by 1 because 0 is total_samples
            results[name] = global_sums[i+1].item() / global_samples
    else:
        for name in metric_names:
            results[name] = 0.0
    
    # Gather and sum token_counts across all processes
    gathered_token_counts = None
    if accumulated_token_counts is not None:
        gathered_token_counts = accelerator.gather(accumulated_token_counts)
        # Sum across all processes
        if gathered_token_counts.dim() > 1:
            gathered_token_counts = gathered_token_counts.sum(dim=0)
            
    return results, gathered_token_counts


def evaluate(
    model: FSQVQVAE,
    dataloaders: Dict[str, DataLoader],
    accelerator: Accelerator,
    max_eval_batches: int | None = None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], list, Dict[str, torch.Tensor]]:
    """
    Evaluate model on validation dataloaders.
    
    Runs on ALL processes in parallel for speed, but returns results only on main process.
    Other processes get empty results.
    
    Returns:
        per_dataset: Dict mapping dataset name to metrics dict
        aggregated: Dict of aggregated metrics across all datasets
        metric_names: List of metric names
        per_dataset_token_counts: Dict mapping dataset name to accumulated token count histograms
    """
    per_dataset: Dict[str, Dict[str, float]] = {}
    per_dataset_token_counts: Dict[str, torch.Tensor] = {}
    
    if not dataloaders:
        return per_dataset, {}, [], per_dataset_token_counts

    model.eval()
    with torch.no_grad():
        # All processes participate in evaluation for speed
        for name, dataloader in dataloaders.items():
            if dataloader is None:
                continue
            metrics, token_counts = _evaluate_single_dataloader(
                model, dataloader, accelerator, max_eval_batches=max_eval_batches
            )
            per_dataset[name] = metrics
            if token_counts is not None:
                per_dataset_token_counts[name] = token_counts
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
    return per_dataset, aggregated, metric_names, per_dataset_token_counts


__all__ = ["evaluate"]
