# VQ-VAE Latent Actions

Training pipeline for FSQ-based VQ-VAE models that learn discrete latent action spaces for robotics policies. The project relies on [Hydra](https://hydra.cc/) for configuration, [Accelerate](https://github.com/huggingface/accelerate) for hardware abstraction, and [ClearML](https://github.com/allegroai/clearml) for experiment tracking.

## Repository layout

```
configs/                # Hydra configs (top-level + model-specific overrides)
src/vqvae_latent_actions/
  models/               # FSQ-VQ-VAE implementation (with save/load helpers)
  training/             # Training loop + evaluation utilities
  utils/                # Logging, metric tracking helpers
train.py                # Hydra entry point
```

`dataset.py` is assumed to provide `prepare_dataloaders(batch_size)` which returns:

1. An example action tensor to infer dimensions.
2. A train dataloader.
3. A dict of validation dataloaders (`{dataset_name: dataloader}`).

Validation runs loop through every dataset-specific dataloader, log per-dataset metrics, and report aggregated validation curves (mean over datasets) to ClearML plus `.png` snapshots under `outputs/<run>/figures/`.

## Running training

```bash
export CLEARML_CONFIG_FILE=/path/to/clearml.conf  # used by ClearML SDK
python train.py trainer.num_steps=20000 dataset.batch_size=256 model.latent_dim=512
```

Hydra lets you override any setting from the CLI. Model-specific knobs live under `configs/model/<name>.yaml`. For example, to change FSQ levels and transformer depth:

```bash
python train.py \
  model.latent_dim=384 \
  model.num_encoder_layers=6 \
  model.num_decoder_layers=6 \
  model.fsq_levels='[9,9,9,9]'
```

### ClearML logging

If `clearml.enable=true` (default) and the `CLEARML_CONFIG_FILE` environment variable points to a valid ClearML credentials file, the script will create a task under the configured project/tag names and push scalars plus validation plots. Set `clearml.enable=false` to disable tracking.

### Accelerate configuration

Accelerate automatically selects devices (CPU/CUDA/MPS) and manages gradient accumulation + mixed precision from `trainer.*` settings. To launch multi-GPU training, follow the usual `accelerate launch` flow:

```bash
accelerate launch --multi_gpu --num_processes=4 train.py trainer.num_steps=50000
```

## Using the model

After (or during) training, checkpoints land in `outputs/<run>/checkpoints/`. Each checkpoint contains `pytorch_model.bin` and `config.json`. You can reload weights via:

```python
from vqvae_latent_actions.models import FSQVQVAE

model = FSQVQVAE.from_pretrained("outputs/2024-05-01_12-00-00/checkpoints/best_step_10000")
model.eval()
```

You can also manually save in code via `model.save_pretrained(path)` or `model.save_state(path)`.

## Notes

- All loss terms are computed inside `FSQVQVAE.compute_loss` to keep the training loop generic.
- Validation averages are plotted with matplotlib and logged to ClearML; the latest plot path is also stored in the `LAST_VAL_FIG` environment variable for quick reference.
- The FSQ quantizer is fully deterministic (no learnable codebooks) and returns token indices for use in downstream discrete planners.
