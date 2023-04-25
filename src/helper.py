import torch

from src.constants import INPUT_DIR
from src.specs import ArgParams

longformer_model_name = "allenai/longformer-base-4096"
nystromformer_model_name = "uw-madison/nystromformer-4096"
reformer_model_name = "robingeibel/reformer-finetuned-big_patent-4096"
bigbird_model_name = "google/bigbird-roberta-base"
bert_model_name = "bert-base-uncased"


def validate(params: ArgParams) -> ArgParams:
    params.device = [int(x) for x in params.device]

    return params


def get_hparams(model_name: str = "bert"):
    hparams = ArgParams(
        run_name="MSMARCO",
        model_name=bert_model_name,
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

    if model_name == "bert":
        hparams.model_name = bert_model_name
        hparams.max_seq_len = 512

    if model_name == "longformer":
        hparams.model_name = longformer_model_name
        hparams.max_seq_len = 4096

    if model_name == "nystromformer":
        hparams.model_name = nystromformer_model_name
        hparams.max_seq_len = 4096

    if model_name == "bigbird":
        hparams.model_name = bigbird_model_name
        hparams.max_seq_len = 4096

    return hparams
