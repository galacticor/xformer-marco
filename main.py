import dataclasses
import sys
import argparse

from src.longformer import main as runner
from src.specs import ArgParams
from src.helper import get_hparams, validate


if __name__ == "__main__":
    args = sys.argv

    hparams = get_hparams()

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

    parser.add_argument("--validation", type=int, default=20,
                        help="Validation Size")

    parser.add_argument("--training", type=int, default=100,
                        help="Training Size")

    parser.add_argument("--test", type=int, default=2048,
                        help="Test Sequence Size")
    
    parser.add_argument("--mode", type=str, default="train",
                        help="Mode: train, test, inference")

    param_from_parser = parser.parse_args()

    hparams.data_loader_bs = param_from_parser.data_loader_bs
    hparams.val_data_loader_bs = param_from_parser.val_data_loader_bs
    hparams.num_workers = param_from_parser.num_workers
    hparams.device = param_from_parser.device
    hparams.ckpt_path = param_from_parser.ckpt_path
    hparams.training = param_from_parser.training
    hparams.validation = param_from_parser.validation

    if hparams.val_data_loader_bs <= 0:
        hparams.val_data_loader_bs = hparams.data_loader_bs

    hparams = validate(hparams)

    hparams: ArgParams = argparse.Namespace(**dataclasses.asdict(hparams))
    print(hparams)
    trainer, model = runner(hparams)

    if hparams.mode.lower() == "train":
        trainer.fit(model, ckpt_path=hparams.ckpt_path)
        trainer.save_checkpoint(f"{hparams.model_name}_{hparams.run_name}_sz{hparams.training}:{hparams.validation}_bs{hparams.data_loader_bs}:{hparams.val_data_loader_bs}_last.ckpt")

    if hparams.mode.lower() == "test":
        trainer.test(model)
