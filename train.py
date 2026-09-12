"""Hydra entry point.

    python train.py run_name=hier-fsq-n10-v2048                       # defaults: FSQ, N=10, V=2048, 300k steps
    python train.py model=hier_vq model.num_tokens=16 train.steps=100000
"""
from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from vqvae_latent_actions.training.loop import TrainConfig, train


def build_train_config(cfg: DictConfig) -> TrainConfig:
    payload = OmegaConf.to_container(cfg, resolve=True)
    train_section = dict(payload["train"])
    data = payload["data"]
    return TrainConfig(manifest=data["manifest"], eval_set=data["eval_set"], cache_dir=data.get("cache_dir"),
                       model=dict(payload["model"]), comet=dict(payload.get("comet", {})), **train_section)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    train(build_train_config(cfg))


if __name__ == "__main__":
    main()
