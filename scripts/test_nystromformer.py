import dataclasses
import sys
import argparse
import os
import json

from src.helper import get_hparams, validate
from src.longformer import main as runner
from src.specs import ArgParams


BASE_DIR_CHECKPOINTS = "/workspace/old_workspace/xformer-marco/xformer-marco/3sizvf8a/checkpoints/uw-madison/"

CHECKPOINTS = [
    "nystromformer-4096_epoch_10-step_17589-val_epoch_loss_0.12.ckpt",
    "nystromformer-4096_epoch_11-step_19188-val_epoch_loss_0.12.ckpt",
    "nystromformer-4096_epoch_12-step_20787-val_epoch_loss_0.12.ckpt",
    "nystromformer-4096_epoch_13-step_22386-val_epoch_loss_0.12.ckpt",
    "nystromformer-4096_epoch_14-step_23985-val_epoch_loss_0.12.ckpt",
    "nystromformer-4096_epoch_5-step_9594-val_epoch_loss_0.12.ckpt",
    "nystromformer-4096_epoch_6-step_11193-val_epoch_loss_0.12.ckpt",
    "nystromformer-4096_epoch_7-step_12792-val_epoch_loss_0.12.ckpt",
    "nystromformer-4096_epoch_8-step_14391-val_epoch_loss_0.12.ckpt",
    "nystromformer-4096_epoch_9-step_15990-val_epoch_loss_0.12.ckpt",
]


def main(params: ArgParams):
    print(params)
    trainer, model = runner(params)
    model.update_test(params.test)

    results = {}

    for ckpt in CHECKPOINTS:
        print(ckpt)
        ckpt_path = os.path.join(BASE_DIR_CHECKPOINTS, ckpt),
        result = trainer.test(model, ckpt_path=ckpt_path)
        results[ckpt] = result

    print(results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    args = sys.argv

    hparams = get_hparams(model_name="nystromformer")

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

    parser.add_argument("--test", type=str, default="2048",
                        help="Test Sequence Dataset")

    param_from_parser = parser.parse_args()
    hparams.data_loader_bs = param_from_parser.data_loader_bs
    hparams.val_data_loader_bs = param_from_parser.val_data_loader_bs
    hparams.num_workers = param_from_parser.num_workers
    hparams.device = param_from_parser.device
    hparams.ckpt_path = param_from_parser.ckpt_path
    hparams.training = param_from_parser.training
    hparams.validation = param_from_parser.validation
    hparams.test = param_from_parser.test

    if hparams.val_data_loader_bs <= 0:
        hparams.val_data_loader_bs = hparams.data_loader_bs

    hparams = validate(hparams)

    hparams: ArgParams = argparse.Namespace(**dataclasses.asdict(hparams))

    main(hparams)
