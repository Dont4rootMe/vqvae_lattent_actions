export WANDB_MODE=offline 
export CLEARML_CONFIG_FILE='/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/clearml_config.conf'

accelerate launch --config_file /mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/accelerate_config.yaml \
    train.py \
    model=fsq_vqvae \
    trainer.num_steps=100000 \
    dataset.batch_size=32 \
    model.latent_dim=512 \
    dataset.batch_size=256 \
    trainer.log_interval=5 \
    trainer.val_interval=200 \
    trainer.save_interval=1000 \
    trainer.gradient_accumulation_steps=1 \
    trainer.mixed_precision="bf16" \
    experiment_name=fsq_vqvae_initial
