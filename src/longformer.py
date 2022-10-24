import pytorch_lightning as pl

from .specs import ArgParams
from .TransformerMarco import TransformerMarco

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_lightning import seed_everything

seed_everything(42)


def main(hparams: ArgParams, model=None):
    if model is None:
        model = TransformerMarco(hparams)

    loggers = []
    if hparams.use_wandb:
        wandb_logger = WandbLogger(
            entity="widyantohadi",
            name=f"{hparams.model_name}_{hparams.run_name}_bs{hparams.data_loader_bs}:{hparams.val_data_loader_bs}",
        )
        wandb_logger.log_hyperparams(hparams)
        loggers.append(wandb_logger)
    if hparams.use_tensorboard:
        tb_logger = TensorBoardLogger(
            "tb_logs", name=f"Longformer-docs", version=hparams.slurm_job_id
        )
        loggers.append(tb_logger)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=hparams.epochs,
        verbose=True,
        monitor="val_epoch_loss",
        mode="min",
        filename=hparams.model_name + '_{epoch}-{step}-{val_epoch_loss:.2f}'
    )

    # This Trainer handles most of the stuff.
    # Enables distributed training with one line:
    # https://towardsdatascience.com/trivial-multi-node-training-with-pytorch-lightning-ff75dfb809bd
    trainer = pl.Trainer(
        devices=hparams.device,
        num_nodes=hparams.num_nodes,
        resume_from_checkpoint=hparams.ckpt_path,
        # distributed_backend=hparams.distributed_backend,
        # control the effective batch size with this param
        accumulate_grad_batches=hparams.trainer_batch_size,
        # Training will stop if max_steps or max_epochs have reached (earliest).
        max_epochs=hparams.epochs,
        # max_steps=hparams.num_training_steps,
        logger=loggers,
        callbacks=[checkpoint_callback],
        strategy=None if hparams.accelerator == "cpu" else "dp",
        accelerator=hparams.accelerator,
        # progress_bar_callback=False,
        # progress_bar_refresh_rate=0,
        # use_amp=True --> use 16bit precision
        val_check_interval=hparams.val_check_interval,  # val every N steps
        # num_sanity_val_steps=5,
        # fast_dev_run=True,
        precision=16,
        amp_backend="native",
    )
    return trainer, model
