"""Dataset helpers for BEAST tokenizer training scripts."""
from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Tuple

import hydra
import torch
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from lerobot.common.datasets.create_dataloader import create_lerobot_dataset_by_config
from lerobot.common.utils.inference_transforms import get_torch_output_transforms
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from lerobot.common.datasets.torch_transforms import compose

# Default budgets reused by the training scripts.
BEAST_TRAIN_MAX_SAMPLES = 100_000
BPE_TRAIN_MAX_SAMPLES = 50_000
BPE_ERROR_BATCHES = 12_500

os.environ["OPENPI_DATA_HOME"] = (
    "/mnt/virtual_ai0001071-01239_SR006-nfs2/apanasevich/openpi/assets"
)
os.environ["HF_HOME"] = "/mnt/virtual_ai0001071-01239_SR006-nfs2/.cache/huggingface"
os.environ["XDG_CACHE_HOME"] = "/mnt/virtual_ai0001071-01239_SR006-nfs2/.cache"


def create_train_val_datasets_distributed(
    data_config_factory: Any,
    model_config: Any,
    assets_dirs: str,
    accelerator: Accelerator,
    val_split: float = 0.1,
    seed: int = 42,
    map_to_unified_space: bool = False,
    use_validation_list: bool = False,
    recompute_norm_stats: bool = False,
    **dataset_kwargs,
):
    """Replica of the dataset splitter used by the research scripts."""

    class DummyFactory:
        def __init__(self, cfg_item):
            self.cfg_item = cfg_item

        def create(self, *args, **kwargs):
            return self.cfg_item

    data_cfg = data_config_factory.create(assets_dirs, model_config)
    if hasattr(data_cfg, "mixture_configs") and data_cfg.mixture_configs:
        dataset_configs = data_cfg.mixture_configs
        dataset_names = [cfg_item.repo_id for cfg_item in dataset_configs]
    else:
        dataset_configs = [data_cfg]
        dataset_names = ["main"]

    if accelerator.is_main_process:
        train_episodes_dict = {}
        val_episodes_dict = {}
        validation_episodes_loaded = False

        for i, cfg_item in enumerate(dataset_configs):
            cfg_name = dataset_names[i]
            if use_validation_list and hasattr(cfg_item, "validation_episodes") and cfg_item.validation_episodes:
                validation_episodes_path = cfg_item.validation_episodes
                if os.path.exists(validation_episodes_path):
                    try:
                        with open(validation_episodes_path, "r") as f:
                            validation_episodes_data = json.load(f)

                        if isinstance(validation_episodes_data, list):
                            val_episodes_dict[cfg_name] = validation_episodes_data
                        elif isinstance(validation_episodes_data, dict):
                            val_episodes_dict[cfg_name] = validation_episodes_data.get(cfg_name)
                        else:
                            val_episodes_dict[cfg_name] = None

                        if val_episodes_dict[cfg_name] is not None:
                            validation_episodes_loaded = True
                            logging.info(
                                "Loaded validation episodes for %s from %s: %s",
                                cfg_name,
                                validation_episodes_path,
                                val_episodes_dict[cfg_name],
                            )
                        else:
                            logging.warning(
                                "Could not extract validation episodes for %s from %s",
                                cfg_name,
                                validation_episodes_path,
                            )
                    except Exception as exc:
                        logging.warning(
                            "Failed to load validation episodes for %s from %s: %s",
                            cfg_name,
                            validation_episodes_path,
                            exc,
                        )
                        val_episodes_dict[cfg_name] = None
                else:
                    logging.warning(
                        "Validation episodes file %s not found for %s",
                        validation_episodes_path,
                        cfg_name,
                    )
                    val_episodes_dict[cfg_name] = None
            else:
                val_episodes_dict[cfg_name] = None

        if not validation_episodes_loaded:
            logging.warning(
                "No validation episode files found or use_validation_list=False, using random splitting",
            )
            for i, cfg_item in enumerate(dataset_configs):
                cfg_name = dataset_names[i]
                info_path = os.path.join(cfg_item.root_dir, "meta", "info.json")
                with open(info_path, "r") as f:
                    total_episodes = json.load(f)["total_episodes"]

                all_episodes = list(range(total_episodes))
                random.seed(seed + hash(cfg_name))
                random.shuffle(all_episodes)

                n_val = max(1, int(total_episodes * val_split))
                val_episodes = all_episodes[:n_val]
                train_episodes = all_episodes[n_val:]

                val_episodes_dict[cfg_name] = val_episodes
                train_episodes_dict[cfg_name] = train_episodes
        else:
            for i, cfg_item in enumerate(dataset_configs):
                cfg_name = dataset_names[i]
                info_path = os.path.join(cfg_item.root_dir, "meta", "info.json")
                with open(info_path, "r") as f:
                    total_episodes = json.load(f)["total_episodes"]

                if val_episodes_dict[cfg_name] is not None:
                    all_episodes = set(range(total_episodes))
                    val_episodes_set = set(val_episodes_dict[cfg_name])
                    train_episodes = sorted(list(all_episodes - val_episodes_set))
                    train_episodes_dict[cfg_name] = train_episodes
                else:
                    all_episodes = list(range(total_episodes))
                    random.seed(seed + hash(cfg_name))
                    random.shuffle(all_episodes)

                    n_val = max(1, int(total_episodes * val_split))
                    val_episodes = all_episodes[:n_val]
                    train_episodes = all_episodes[n_val:]

                    val_episodes_dict[cfg_name] = val_episodes
                    train_episodes_dict[cfg_name] = train_episodes
    else:
        train_episodes_dict = None
        val_episodes_dict = None

    train_episodes_dict = broadcast_object_list([train_episodes_dict])[0]
    val_episodes_dict = broadcast_object_list([val_episodes_dict])[0]

    data_cfg = data_config_factory.create(assets_dirs, model_config)
    is_mixture_cfg = hasattr(data_cfg, "mixture_configs") and data_cfg.mixture_configs

    cfg_map = {}
    if is_mixture_cfg:
        for item in data_cfg.mixture_configs:
            cfg_map[item.repo_id] = item
    else:
        cfg_map["main"] = data_cfg

    allowed_by_cfg = {}
    for name, cfg_item in cfg_map.items():
        allowlist_path = getattr(cfg_item, "episodes_list_file", None)
        allowed_set = None
        try:
            if allowlist_path and os.path.exists(allowlist_path):
                with open(allowlist_path, "r") as f:
                    allowed_set = set(json.load(f))
        except Exception:
            allowed_set = None
        allowed_by_cfg[name] = allowed_set

    def _apply_allowlist(eps_dict: dict[str, list[int]]):
        out = {}
        for name, eps in eps_dict.items():
            allowed = allowed_by_cfg.get(name)
            if isinstance(eps, list) and allowed is not None:
                filtered = [e for e in eps if e in allowed]
                if len(filtered) == 0 and len(allowed) > 0:
                    filtered = sorted(list(allowed))
                elif len(allowed) == 0:
                    raise RuntimeError(
                        f"Validation episodes are empty for dataset {name}. Check the provided lists."
                    )
                out[name] = filtered
            else:
                out[name] = eps
        return out

    train_episodes_dict = _apply_allowlist(train_episodes_dict)
    val_episodes_dict = _apply_allowlist(val_episodes_dict)

    if hasattr(data_cfg, "mixture_configs") and data_cfg.mixture_configs:
        world_size = accelerator.num_processes
        rank = accelerator.process_index
        val_episodes_dict_local = {}
        for cfg_name, eps in val_episodes_dict.items():
            if isinstance(eps, list):
                if len(eps) >= world_size:
                    shard = eps[rank::world_size]
                elif len(eps) > 0:
                    shard = [eps[rank % len(eps)]]
                else:
                    shard = []
                val_episodes_dict_local[cfg_name] = shard
            else:
                val_episodes_dict_local[cfg_name] = eps

        train_episodes_dict_local = {}
        for cfg_name, eps in train_episodes_dict.items():
            if isinstance(eps, list):
                if len(eps) >= world_size:
                    shard = eps[rank::world_size]
                elif len(eps) > 0:
                    shard = [eps[rank % len(eps)]]
                else:
                    shard = []
                train_episodes_dict_local[cfg_name] = shard
            else:
                train_episodes_dict_local[cfg_name] = eps
    else:
        train_episodes_dict_local = train_episodes_dict
        val_episodes_dict_local = val_episodes_dict

    train_dataset, norm_stats = create_lerobot_dataset_by_config(
        data_config_factory=data_config_factory,
        model_config=model_config,
        assets_dirs=assets_dirs,
        episodes=train_episodes_dict_local,
        normalization_mode=model_config.normalization_mode,
        return_norm_stats=True,
        map_to_unified_space=map_to_unified_space,
        recompute_norm_stats=recompute_norm_stats,
        **dataset_kwargs,
    )

    if hasattr(train_dataset, "set_rng"):
        try:
            import numpy as _np

            train_dataset.set_rng(_np.random.RandomState(seed + accelerator.process_index))
        except Exception:
            pass

    val_datasets_dict = {}
    output_pipeline_dict = {}
    is_mixture = hasattr(data_cfg, "mixture_configs") and data_cfg.mixture_configs
    if is_mixture:
        cfg_items = {cfg.repo_id: cfg for cfg in data_cfg.mixture_configs}
    else:
        cfg_items = {"main": data_cfg}

    for cfg_name, eps in val_episodes_dict_local.items():
        cfg_item = cfg_items[cfg_name]
        norm_stats_item = norm_stats[cfg_item.repo_id]
        factory = DummyFactory(cfg_item)
        val_dataset = create_lerobot_dataset_by_config(
            data_config_factory=factory,
            model_config=model_config,
            assets_dirs=assets_dirs,
            episodes=eps,
            normalization_mode=model_config.normalization_mode,
            return_norm_stats=False,
            recompute_norm_stats=False,
            precomputed_norm_stats=norm_stats,
            map_to_unified_space=map_to_unified_space,
            **dataset_kwargs,
        )
        val_datasets_dict[cfg_name] = val_dataset
        output_pipeline_dict[cfg_name] = compose(
            get_torch_output_transforms(
                norm_stats=norm_stats_item,
                policy_config=model_config,
                data_config_factory=factory,
                assets_dirs=assets_dirs,
                normalization_mode=model_config.normalization_mode,
                map_to_unified_space=map_to_unified_space,
            )
        )

    return train_dataset, val_datasets_dict, norm_stats, output_pipeline_dict


def instantiate_data_config(cfg: DictConfig, add_kwargs: dict | None = None):
    """Instantiate robotics dataset config with optional overrides."""
    try:
        is_mixture = cfg._target_.split(".")[-1] == "MixtureDataConfigFactory"
    except Exception:
        is_mixture = False

    if is_mixture and hasattr(cfg, "datasets_with_weights") and cfg.datasets_with_weights is not None:
        assert cfg.data_configs is None, "both datasets_with_weights and data_configs are set"
        assert cfg.weights is None, "both datasets_with_weights and weights are set"
        datasets_list = [ds_cfg.path for ds_cfg in cfg.datasets_with_weights]
        weights_list = [ds_cfg.weight for ds_cfg in cfg.datasets_with_weights]
        cfg.data_configs = datasets_list
        cfg.weights = weights_list
        del cfg.datasets_with_weights

    if add_kwargs:
        if is_mixture and hasattr(cfg, "data_configs") and cfg.data_configs is not None:
            for idx in range(len(cfg.data_configs)):
                dc = cfg.data_configs[idx]
                try:
                    for k, v in add_kwargs.items():
                        if isinstance(dc, DictConfig):
                            if k in dc:
                                dc[k] = v
                        elif isinstance(dc, dict):
                            if k in dc:
                                dc[k] = v
                except Exception:
                    pass
        else:
            for k, v in add_kwargs.items():
                try:
                    if k in cfg:
                        cfg[k] = v
                except Exception:
                    pass

    if is_mixture:
        return hydra.utils.instantiate(cfg, _recursive_=False)
    else:
        return hydra.utils.instantiate(cfg, _recursive_=True)


def get_datasets() -> Tuple[Any, Any, Any, Any]:
    """Load robotics datasets using the configuration checkpoint."""
    try:
        OmegaConf.register_resolver(
            "_load_config", lambda rel_path: OmegaConf.load(os.path.join(os.getcwd(), rel_path))
        )
    except Exception:
        pass

    accelerator = Accelerator()
    assets_dir = "/mnt/virtual_ai0001071-01239_SR006-nfs2/apanasevich/pi0_assets_v4"
    cfg = torch.load("config.ckpt", weights_only=False)
    
    cfg.robotics_dataset.data_configs = cfg.robotics_dataset.data_configs[:-1]
    cfg.robotics_dataset.weights = cfg.robotics_dataset.weights[:-1]

    map_to_unified_space = False
    map_to_humanoid = False
    add_kwargs = {
        "map_to_unified_space": map_to_unified_space,
        "map_to_humanoid": map_to_humanoid,
    }
    robotics_dataset_factory = instantiate_data_config(cfg.robotics_dataset, add_kwargs)
    policy_config = hydra.utils.instantiate(cfg.policy.policy_config)

    val_split = 0.05

    robotics_dataset, val_datasets_dict, norm_stats, output_pipeline_dict = (
        create_train_val_datasets_distributed(
            data_config_factory=robotics_dataset_factory,
            model_config=policy_config,
            assets_dirs=assets_dir,
            accelerator=accelerator,
            val_split=val_split,
            seed=42,
            map_to_unified_space=map_to_unified_space,
            use_validation_list=True,
            recompute_norm_stats=False,
        )
    )

    return robotics_dataset, val_datasets_dict, norm_stats, output_pipeline_dict


def prepare_dataloaders(batch_size: int) -> Tuple[Any, DataLoader]:
    """Return the robotics dataset and a DataLoader ready for tokenizer training."""
    robotics_dataset, val_datasets_dict, _, _ = get_datasets()
    
    def _create_dataloader(dataset) -> DataLoader:
        dtl = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        try:
            dtl.dataset._dataset._dataset.return_fake_images = True
        except Exception:
            for d in dtl.dataset._datasets:
                d._dataset._dataset.return_fake_images = True
        
        return dtl
    
    dataloader_train = _create_dataloader(robotics_dataset)
    dataloader_evals = {dts_name: _create_dataloader(dts) for dts_name, dts in val_datasets_dict.items()}
    
    example_actions = robotics_dataset[0]['actions']
    return example_actions, dataloader_train, dataloader_evals
