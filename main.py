import dataclasses
import sys
import torch
import argparse

from src import longformer
from src.constants import DATA_DIR
from src.specs import ArgParams


if __name__ == "__main__":
    args = sys.argv
    hparams: ArgParams

    if args[1] == "longformer":
        runner = longformer.main
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
            device=0 if torch.cuda.is_available() else -1,
            gpus=1 if torch.cuda.is_available() else 0,
            num_nodes=1,
        )


    if hparams.val_data_loader_bs <= 0:
        hparams.val_data_loader_bs = hparams.data_loader_bs

    hparams = argparse.Namespace(**dataclasses.asdict(hparams))
    print(hparams)
    runner(hparams)
