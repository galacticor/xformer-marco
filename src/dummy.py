import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer-MARCO")
    parser.add_argument("--gpus", type=int, default=1, help="Num of GPUs per node")
    hparams = parser.parse_args()
    print(hparams)
    argparse.Namespace