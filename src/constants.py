import os

INPUT_DIR = os.getenv("INPUT_DIR", "../input")
DOCS_DIR = os.getenv("DOCS_DIR", "../input/docs")
DOCS_DIR_1 = os.getenv("DOCS_DIR", "../input/docs")
DOCS_DIR_2 = os.getenv("DOCS_DIR", "../input/docs")
DOCS_FILE = os.getenv("DOCS_FILE", "../input/docs.tsv")

PROCESSED_DIR = os.getenv("PROCESSED_DIR", "../input/processed-data-msmarco/")
CORPUS_DIR = PROCESSED_DIR + os.getenv("CORPUS_DIR", "corpus_sampled/")
DATA_DIR = PROCESSED_DIR + os.getenv("CORPUS_DIR", "data_sampled/")
TEST_DIR = PROCESSED_DIR + os.getenv("TEST_DIR", "data_test/")
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "../input/latest-epoch.ckpt")

os.environ["WANDB_API_KEY"] = "769a2304e87b5f9fcd2ec42128ddc7b22ce4a617"
