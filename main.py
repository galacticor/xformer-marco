import dataclasses
import sys
import torch
import argparse

from src import longformer
from src.constants import INPUT_DIR
from src.specs import ArgParams


if __name__ == "__main__":
    args = sys.argv

    longformer_model_name = "allenai/longformer-base-4096"
    nystromformer_model_name = "uw-madison/nystromformer-4096"
    reformer_model_name = "robingeibel/reformer-finetuned-big_patent-4096"
    bert_model_name = "bert-base-uncased"

    hparams = ArgParams(
        run_name="MSMARCO50k-20k",
        model_name=bert_model_name,
    #     ckpt_path=CHECKPOINT_PATH,
        ckpt_path=None,
        validation="validation_20k.csv",
        training="training_50k.csv",


        learning_rate=1e-4,
        num_warmup_steps=100,
        num_training_steps=30000,
        data_dir=INPUT_DIR,
        max_seq_len=512,
        data_loader_bs=16,
        val_data_loader_bs=8,
        num_workers=2,
        trainer_batch_size=4,
        epochs=3,
        use_wandb=True,
        use_tensorboard=False,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        val_check_interval=1.0,
        device=2,
        num_nodes=1
    )

    if args[1] == "bert":
        runner = longformer.main
        hparams.model_name = bert_model_name
        hparams.max_seq_len = 512

    if args[1] == "longformer":
        runner = longformer.main
        hparams.model_name = longformer_model_name
        hparams.max_seq_len = 4096

    if args[1] == "nystromformer":
        runner = longformer.main
        hparams.model_name = nystromformer_model_name
        hparams.max_seq_len = 4096


    if hparams.val_data_loader_bs <= 0:
        hparams.val_data_loader_bs = hparams.data_loader_bs

    hparams = argparse.Namespace(**dataclasses.asdict(hparams))
    print(hparams)
    trainer, model = runner(hparams)

    trainer.fit(model)
    trainer.save_checkpoint("latest-epoch.ckpt")
