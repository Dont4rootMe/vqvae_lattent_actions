"""Training loop orchestrator."""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import torch
from accelerate import Accelerator
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from ..models import FSQVQVAE
from ..training.eval import evaluate
from ..utils.logging import MetricLogger
from ..utils.metrics import ValidationCurveTracker

from tqdm import tqdm


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


def _compute_gradient_stats(model: torch.nn.Module) -> Dict[str, float]:
    """Compute gradient statistics for logging."""
    total_norm = 0.0
    num_params = 0
    max_grad = 0.0
    min_grad = float('inf')
    grad_sum = 0.0
    
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            num_params += p.grad.numel()
            grad_sum += p.grad.data.abs().sum().item()
            max_grad = max(max_grad, p.grad.data.abs().max().item())
            min_grad = min(min_grad, p.grad.data.abs().min().item())
    
    total_norm = total_norm ** 0.5
    mean_grad = grad_sum / num_params if num_params > 0 else 0.0
    
    return {
        'grad_norm': total_norm,
        'grad_mean': mean_grad,
        'grad_max': max_grad,
        'grad_min': min_grad if min_grad != float('inf') else 0.0,
    }


def _log_train_metrics(
    accelerator: Accelerator,
    logger: MetricLogger,
    outputs,
    lr: float,
    step: int,
    grad_stats: Optional[Dict[str, float]] = None,
) -> None:
    
    metrics = {'lr': torch.tensor(lr, device=accelerator.device)}
    
    if hasattr(outputs, '__dataclass_fields__'):
        field_names = outputs.__dataclass_fields__.keys()
    else:
        field_names = [name for name in dir(outputs) if not name.startswith("_")]
    
    for name in field_names:
        value = getattr(outputs, name)
        # Only process tensor attributes
        if isinstance(value, torch.Tensor):
            value = value.detach()
            # Only log scalars or 1D tensors
            if value.dim() < 1:
                metrics[name] = value
    
    # Add gradient statistics if provided
    if grad_stats is not None:
        for name, value in grad_stats.items():
            metrics[name] = torch.tensor(value, device=accelerator.device)

    reduced = {}
    for name, value in sorted(metrics.items(), key=lambda x: x[0]):
        tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value, device=accelerator.device)
        gathered = accelerator.gather(tensor.detach())
        reduced[name] = gathered.mean().item()        
    if accelerator.is_main_process:
        logger.log("train", reduced, step)
    

def _save_checkpoint(
    model: FSQVQVAE,
    optimizer: Optimizer,
    scheduler: Optional[LRScheduler],
    accelerator: Accelerator,
    output_dir: Path,
    step: int,
    tag: str,
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        ckpt_dir = output_dir / "checkpoints" / f"{tag}_step_{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(str(ckpt_dir))
        state = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "step": step,
        }
        torch.save(state, ckpt_dir / "trainer_state.pt")


def run_train_loop(
    cfg,
    model: FSQVQVAE,
    optimizer: Optimizer,
    scheduler: Optional[LRScheduler],
    accelerator: Accelerator,
    dataloader_train: DataLoader,
    dataloader_val: Dict[str, DataLoader],
    metric_logger: MetricLogger,
) -> None:
    output_dir = Path(cfg.trainer.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)

    tracker = ValidationCurveTracker()
    global_step = 0
    best_val = math.inf

    max_steps = cfg.trainer.num_steps
    log_interval = cfg.trainer.log_interval
    val_interval = cfg.trainer.val_interval
    save_interval = cfg.trainer.save_interval
    grad_clip = cfg.trainer.max_grad_norm
    
    max_eval_batches = cfg.trainer.max_eval_batches

    train_iterator = iter(dataloader_train)


    pbar = tqdm(total=max_steps, desc="Training", leave=False)
    while global_step < max_steps:
        try:
            batch = next(train_iterator)
            pbar.update(1)
        except StopIteration:
            train_iterator = iter(dataloader_train)
            batch = next(train_iterator)
            pbar.update(1)

        batch = _move_batch_to_device(batch, accelerator.device)
        batch = _ensure_actions_dtype(batch, model)
        
        with accelerator.accumulate(model):
            outputs = model(batch["actions"])
            accelerator.backward(outputs.loss)
            
            # Compute gradient statistics before clipping for accurate measurements
            grad_stats = None
            if accelerator.sync_gradients and global_step % log_interval == 0:
                grad_stats = _compute_gradient_stats(model)
            
            if accelerator.sync_gradients and grad_clip is not None:
                accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                
            optimizer.step()
            optimizer.zero_grad()
            
            if scheduler is not None:
                scheduler.step()

        global_step += 1
        
        if global_step % log_interval == 0 and accelerator.sync_gradients:
            current_lr = optimizer.param_groups[0]["lr"]
            _log_train_metrics(accelerator, metric_logger, outputs, current_lr, global_step, grad_stats)

        if val_interval > 0 and global_step % val_interval == 0:
            # All processes participate in evaluation for speed
            per_dataset_metrics, aggregated_metrics, metric_names, per_dataset_token_counts = evaluate(
                model, dataloader_val, accelerator, max_eval_batches
            )
            
            # Only main process logs and saves
            if accelerator.is_main_process:
                # plot comparision of val metrics on datasets
                for metric_name in metric_names:
                    consolidation = {}
                    for dataset_name, metrics in per_dataset_metrics.items():
                        consolidation[dataset_name] = metrics[metric_name]
                    metric_logger.log(f"val-consolidated/{metric_name}", consolidation, global_step, in_one_praph=True)
                
                # plot sepparately for each dataset
                for dataset_name, metrics in per_dataset_metrics.items():
                    metric_logger.log(f"val-dataset/{dataset_name}", metrics, global_step)
                
                # Log token utilization histograms from accumulated token_counts
                if per_dataset_token_counts:
                    # Histogram for all validation datasets combined
                    all_token_counts = sum(per_dataset_token_counts.values())
                    metric_logger.log_histogram(
                        "token_histogram/all_datasets",
                        "histogram",
                        all_token_counts,
                        global_step
                    )
                    
                    # Histogram for each dataset separately
                    for dataset_name, token_counts in per_dataset_token_counts.items():
                        metric_logger.log_histogram(
                            f"token_histogram/{dataset_name}",
                            "histogram",
                            token_counts,
                            global_step
                        )
                
                # if aggregated_metrics:
                #     metric_logger.log("val-aggregation", aggregated_metrics, global_step)
                #     tracker.update(global_step, aggregated_metrics)
                    
                #     fig = tracker.plot()
                #     fig_path = plot_dir / f"val_metrics_step_{global_step}.png"
                #     fig.savefig(fig_path)
                #     metric_logger.log_figure("validation", f"step_{global_step}", fig, global_step)
                #     os.environ["LAST_VAL_FIG"] = str(fig_path)
                #     plt.close(fig)
                # if aggregated_metrics and aggregated_metrics.get("loss", math.inf) < best_val:
                #     best_val = aggregated_metrics["loss"]
                #     _save_checkpoint(
                #         model,
                #         optimizer,
                #         scheduler,
                #         accelerator,
                #         output_dir,
                #         global_step,
                #         tag="best",
                #     )
            
            # Synchronize all processes after validation
            accelerator.wait_for_everyone()

        if save_interval > 0 and global_step % save_interval == 0:
            _save_checkpoint(
                model,
                optimizer,
                scheduler,
                accelerator,
                output_dir,
                global_step,
                tag="ckpt",
            )

        if accelerator.sync_gradients and global_step >= max_steps:
            break

    _save_checkpoint(
        model,
        optimizer,
        scheduler,
        accelerator,
        output_dir,
        global_step,
        tag="final",
    )
