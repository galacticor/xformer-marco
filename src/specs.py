from dataclasses import dataclass, field
from typing import List


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
    devices: int = 1
    use_10_percent_of_dev: bool = False
    epochs: int = 1
    learning_rate: float = 3e-5
    num_warmup_steps: int = 250
    num_training_steps: int = 1200
    val_check_interval: float = 0.25
    use_wandb: bool = True
    use_tensorboard: bool = True
    gpus: List[int] = field(default_factory=lambda: [0])
    num_nodes: int = 1
    distributed_backend: str = "dp"
    slurm_job_id: str = "ssss"
    msmarco_ver: str = "2022"
