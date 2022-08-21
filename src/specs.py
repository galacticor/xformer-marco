from dataclasses import dataclass


@dataclass
class ArgParams:
    run_name: str
    model_name: str
    data_dir: str
    max_seq_len: int
    data_loader_bs: int
    val_data_loader_bs: int
    num_workers: int
    trainer_batch_size: int
    epochs: int = 2
    learning_rate: float = 3e-5
    num_warmup_steps: int = 2500
    num_training_steps: int = 120000
    use_wandb: bool = True
    use_tensorboard: bool = True
    gpus: int = 1
    num_nodes: int = 1
    distributed_backend: str = "dp"
