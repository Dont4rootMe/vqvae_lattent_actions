"""Hydra entry point for FSQ-VQ-VAE training."""
from __future__ import annotations

import math
from typing import Dict, Tuple

import hydra
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

from dataset import prepare_dataloaders
from vqvae_latent_actions.models import FSQVQVAE
from vqvae_latent_actions.training.loop import run_train_loop
from vqvae_latent_actions.utils.logging import MetricLogger, maybe_create_clearml_logger


def _instantiate_model(cfg: DictConfig, action_dim: int) -> FSQVQVAE:
    model_params: Dict = OmegaConf.to_container(cfg.model, resolve=True)
    target = model_params.pop("_target_", "vqvae_latent_actions.models.FSQVQVAE")
    if not model_params.get("action_dim"):
        model_params["action_dim"] = action_dim
    model_params.pop("config", None)
    config_copy = dict(model_params)
    ModelCls = get_class(target)
    return ModelCls(**model_params, config=config_copy)


def _build_optimizer(model: torch.nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        betas=tuple(cfg.optimizer.betas),
        weight_decay=cfg.optimizer.weight_decay,
    )


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    total_steps = cfg.scheduler.total_steps
    if total_steps is None or total_steps <= 0:
        return None
    warmup = cfg.scheduler.warmup_steps
    min_lr = cfg.scheduler.min_lr
    base_lr = cfg.optimizer.lr
    min_factor = min_lr / base_lr if base_lr > 0 else 0.0

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor + (1 - min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _prepare_eval_dataloaders(
    accelerator: Accelerator,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
) -> Dict[str, torch.utils.data.DataLoader]:
    prepared = {}
    for name, dataloader in dataloaders.items():
        prepared[name] = accelerator.prepare_data_loader(dataloader)
    return prepared


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
        mixed_precision=cfg.trainer.mixed_precision,
    )

    if cfg.get("seed") is not None:
        set_seed(cfg.seed)

    accelerator.print("Preparing dataloaders...")
    example_actions, dataloader_train, dataloader_evals = prepare_dataloaders(cfg.dataset.batch_size)
    action_dim = example_actions.shape[-1]

    model = _instantiate_model(cfg, action_dim)
    optimizer = _build_optimizer(model, cfg)

    model, optimizer, dataloader_train = accelerator.prepare(model, optimizer, dataloader_train)
    dataloader_evals = _prepare_eval_dataloaders(accelerator, dataloader_evals)

    scheduler = _build_scheduler(optimizer, cfg)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    clearml_logger = maybe_create_clearml_logger(cfg_dict, accelerator)
    metric_logger = MetricLogger(accelerator, clearml_logger)

    run_train_loop(
        cfg,
        model,
        optimizer,
        scheduler,
        accelerator,
        dataloader_train,
        dataloader_evals,
        metric_logger,
    )


if __name__ == "__main__":
    main()
