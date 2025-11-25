export WANDB_MODE=offline 
export CLEARML_CONFIG_FILE='/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/clearml_config.conf'

accelerate launch --config_file /mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/accelerate_config.yaml \
    train.py \
    model=lfq_vqvae \
    trainer.num_steps=100000 \
    model.latent_dim=256 \
    model.hidden_dim=256 \
    dataset.batch_size=256 \
    trainer.log_interval=10 \
    trainer.val_interval=500 \
    trainer.max_eval_batches=null \
    trainer.save_interval=5000 \
    trainer.gradient_accumulation_steps=1 \
    trainer.mixed_precision="bf16" \
    experiment_name=lfq_Bigger-Quant-Number
