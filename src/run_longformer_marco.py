import dataclasses
import pytorch_lightning as pl
import torch

import argparse
from TransformerMarco import TransformerMarco
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
import os

from pytorch_lightning import seed_everything
from constants import DATA_DIR

from specs import ArgParams

seed_everything(42)


def main(hparams):
    model = TransformerMarco(hparams)

    loggers = []
    if hparams.use_wandb:
        wandb_logger = WandbLogger(
            entity="galacticor",
            name=f"Albert-passage-{hparams.slurm_job_id}",
        )
        wandb_logger.log_hyperparams(hparams)
        loggers.append(wandb_logger)
    if hparams.use_tensorboard:
        tb_logger = TensorBoardLogger(
            "tb_logs", name=f"Longformer-docs", version=hparams.slurm_job_id
        )
        loggers.append(tb_logger)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        verbose=True,
        monitor="val_epoch_loss",
        mode="min",
    )

    # This Trainer handles most of the stuff.
    # Enables distributed training with one line:
    # https://towardsdatascience.com/trivial-multi-node-training-with-pytorch-lightning-ff75dfb809bd
    trainer = pl.Trainer(
        devices=hparams.device,
        num_nodes=hparams.num_nodes,
#         distributed_backend=hparams.distributed_backend,
        # control the effective batch size with this param
        accumulate_grad_batches=hparams.trainer_batch_size,
        # Training will stop if max_steps or max_epochs have reached (earliest).
        max_epochs=hparams.epochs,
        max_steps=hparams.num_training_steps,
        logger=loggers,
        checkpoint_callback=checkpoint_callback,
        # progress_bar_callback=False,
        # progress_bar_refresh_rate=0,
        # use_amp=True --> use 16bit precision
        # val_check_interval=0.25, # val 4 times during 1 train epoch
        val_check_interval=hparams.val_check_interval,  # val every N steps
        # num_sanity_val_steps=5,
        # fast_dev_run=True
    )
    trainer.fit(model)


if __name__ == "__main__":

    hparams = ArgParams(
        run_name="test-1",
        model_name="allenai/longformer-base-4096",
        learning_rate=3e-5,
        num_warmup_steps=1000,
        num_training_steps=120000,
        data_dir=DATA_DIR,
        max_seq_len=4096,
        data_loader_bs=4,
        val_data_loader_bs=2,
        num_workers=1,
        trainer_batch_size=1,
        epochs=1,
        use_wandb=False,
        use_tensorboard=False,
        device=0 if torch.cuda.is_available() else 1,
    #     gpus=1,
        num_nodes=1,
    )

    # hparams = parser.parse_args()
    if hparams.val_data_loader_bs <= 0:
        hparams.val_data_loader_bs = hparams.data_loader_bs

    hparams = argparse.Namespace(**dataclasses.asdict(hparams))
    print(hparams)
    main(hparams)
