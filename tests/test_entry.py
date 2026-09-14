import json
from pathlib import Path

import pytest

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent


def test_config_tree_composes_and_matches_the_budget():
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=["run_name=unit"])
    payload = OmegaConf.to_container(cfg, resolve=True)
    assert payload["model"]["num_tokens"] == 10
    quantizer = payload["model"]["quantizer"]                            # the default arm is the learned codebook
    assert quantizer["type"] == "vq" and quantizer["vocab_size"] == 2048 and quantizer["code_dim"] == 4
    assert quantizer["cosine"] is True                                   # euclidean matching runs away in scale
    assert payload["train"]["steps"] == 300000 and payload["train"]["mixed_precision"] == "bf16"
    assert payload["comet"]["project"] and payload["comet"]["workspace"] == "dont4rootme"
    assert "manifest" in payload["data"] and "eval_set" in payload["data"]


def test_overrides_reach_the_train_config(tmp_path, tiny_manifest, tiny_eval_set):
    from vqvae_latent_actions.data.chunks import save_eval_set
    from train import build_train_config
    eval_path = tmp_path / "eval.npz"
    save_eval_set(tiny_eval_set, eval_path)
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=[
            "run_name=unit", "model=hier_vq", "model.num_tokens=6", "train.steps=7", "train.batch_size=4",
            f"train.out_dir={tmp_path / 'run'}", f"data.manifest={tiny_manifest}", f"data.eval_set={eval_path}",
            "data.cache_dir=null", "comet.mode=disabled"])
    train_cfg = build_train_config(cfg)
    assert train_cfg.steps == 7 and train_cfg.batch_size == 4 and train_cfg.run_name == "unit"
    assert train_cfg.model["num_tokens"] == 6 and train_cfg.model["quantizer"]["type"] == "vq"
    assert train_cfg.comet["mode"] == "disabled" and train_cfg.cache_dir is None
    assert json.loads(json.dumps(train_cfg.model))                       # plain containers, no omegaconf leftovers


@pytest.mark.parametrize("name", ["hier_fsq", "hier_vq", "hier_lfq", "hier_gumbel"])
def test_every_shipped_model_config_builds_its_quantizer(name):
    """Every arm in configs/model must be launchable. The learned-codebook arms were unrunnable for a while:
    they inherited FSQ's `levels`, which no other quantizer accepts."""
    if name == "hier_lfq":
        pytest.importorskip("vector_quantize_pytorch")
    from vqvae_latent_actions.models.quantizers import build_quantizer
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=["run_name=unit", f"model={name}"])
    payload = OmegaConf.to_container(cfg, resolve=True)
    quantizer = build_quantizer(payload["model"]["quantizer"])
    assert quantizer.vocab_size == 2048          # the spec caps the alphabet at FAST's 2048
    assert quantizer.code_dim > 0 and quantizer.out_dim > 0


@pytest.mark.parametrize("overrides, cosine, squash", [
    (["model.quantizer.cosine=true"], True, False),
    (["model.quantizer.cosine=false"], False, False),
    # squash replaces the default cosine anchor, so it has to switch cosine off: the two together are refused
    (["model.quantizer.cosine=false", "model.quantizer.squash=true"], False, True),
])
def test_scale_anchor_overrides_compose_and_build(overrides, cosine, squash):
    """The first cosine check died at composition: `cosine` was not a key of the config, so Hydra refused the
    override before training started."""
    from vqvae_latent_actions.models.quantizers import build_quantizer
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=["run_name=unit", "model=hier_vq", *overrides])
    quantizer = build_quantizer(OmegaConf.to_container(cfg.model.quantizer, resolve=True))
    assert (quantizer.cosine, quantizer.squash) == (cosine, squash)


def test_weight_decay_exemptions_compose():
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=["run_name=unit", "train.no_decay_keywords=[from_code,to_code]"])
    assert OmegaConf.to_container(cfg.train, resolve=True)["no_decay_keywords"] == ["from_code", "to_code"]
