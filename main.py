import dataclasses
import sys
import torch
import argparse

from src import longformer
from src.constants import INPUT_DIR
from src.specs import ArgParams


def validation(params: ArgParams) -> ArgParams:
    params.device = [int(x) for x in params.device]

    return params


if __name__ == "__main__":
    args = sys.argv

    longformer_model_name = "allenai/longformer-base-4096"
    nystromformer_model_name = "uw-madison/nystromformer-4096"
    reformer_model_name = "robingeibel/reformer-finetuned-big_patent-4096"
    bert_model_name = "bert-base-uncased"

    hparams = ArgParams(
        run_name="MSMARCO",
        model_name=bert_model_name,
    #     ckpt_path=CHECKPOINT_PATH,
        ckpt_path=None,
        validation=20,
        training=50,


        learning_rate=1e-4,
        num_warmup_steps=100,
        num_training_steps=30000,
        data_dir=INPUT_DIR,
        max_seq_len=512,
        data_loader_bs=8,
        val_data_loader_bs=4,
        num_workers=32,
        trainer_batch_size=16,
        epochs=10,
        use_wandb=True,
        use_tensorboard=False,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        val_check_interval=1.0,
        device=2,
        num_nodes=1
    )

    parser = argparse.ArgumentParser(description='Transformer-MARCO')
    
    parser.add_argument("--model_name", type=str, default="bert",
                        help="Model name")

    parser.add_argument("--data_loader_bs", type=int, default=8,
                        help="Training batch size")

    parser.add_argument("--val_data_loader_bs", type=int, default=4,
                        help="Validation batch size")

    parser.add_argument("--num_workers", type=int, default=12,
                        help="Num subprocesses for DataLoader")

    parser.add_argument("--device", type=list, default=1,
                        help="GPU Device ID")

    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Checkpoint Path")

    param_from_parser = parser.parse_args()

    if param_from_parser.model_name == "bert":
        runner = longformer.main
        hparams.model_name = bert_model_name
        hparams.max_seq_len = 512

    if param_from_parser.model_name == "longformer":
        runner = longformer.main
        hparams.model_name = longformer_model_name
        hparams.max_seq_len = 4096

    if param_from_parser.model_name == "nystromformer":
        runner = longformer.main
        hparams.model_name = nystromformer_model_name
        hparams.max_seq_len = 4096


    if hparams.val_data_loader_bs <= 0:
        hparams.val_data_loader_bs = hparams.data_loader_bs

    hparams.data_loader_bs = param_from_parser.data_loader_bs
    hparams.val_data_loader_bs = param_from_parser.val_data_loader_bs
    hparams.num_workers = param_from_parser.num_workers
    hparams.device = param_from_parser.device
    hparams.ckpt_path = param_from_parser.ckpt_path

    hparams = validation(hparams)

    hparams = argparse.Namespace(**dataclasses.asdict(hparams))
    print(hparams)
    trainer, model = runner(hparams)

    trainer.fit(model)
    trainer.save_checkpoint(f"{hparams.model_name}_{hparams.run_name}_sz{hparams.training}:{hparams.validation}_bs{hparams.data_loader_bs}:{hparams.val_data_loader_bs}_last.ckpt")
