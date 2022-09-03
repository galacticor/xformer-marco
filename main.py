import dataclasses
import sys
import torch
import argparse

from src import longformer
from src.constants import DATA_DIR
from src.specs import ArgParams


if __name__ == "__main__":
    args = sys.argv
    hparams = ArgParams(
        run_name="test-1",
        model_name="allenai/longformer-base-4096",
        learning_rate=3e-5,
        num_warmup_steps=100,
        num_training_steps=1200,
        data_dir=DATA_DIR,
        max_seq_len=4096,
        data_loader_bs=2,
        val_data_loader_bs=1,
        num_workers=1,
        trainer_batch_size=1,
        epochs=1,
        use_wandb=False,
        use_tensorboard=False,
        # devices=2 if torch.cuda.is_available() else 1,
        gpus=[3] if torch.cuda.is_available() else None,
        num_nodes=2,
        msmarco_ver="2022",
    )

    if args[1] == "longformer":
        runner = longformer.main
        hparams.model_name = "allenai/longformer-base-4096"

    if args[1] == "nystromformer":
        runner = longformer.main
        hparams.model_name = "uw-madison/nystromformer-4096"


    if hparams.val_data_loader_bs <= 0:
        hparams.val_data_loader_bs = hparams.data_loader_bs

    hparams = argparse.Namespace(**dataclasses.asdict(hparams))
    print(hparams)
    runner(hparams)
