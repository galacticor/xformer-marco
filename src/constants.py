import os

INPUT_DIR = os.getenv("INPUT_DIR", "../input")
DOCS_DIR = os.getenv("DOCS_DIR", "../input/docs")
DOCS_DIR_1 = os.getenv("DOCS_DIR", "../input/docs")
DOCS_DIR_2 = os.getenv("DOCS_DIR", "../input/docs")
DOCS_FILE = os.getenv("DOCS_FILE", "../input/docs.tsv")

PROCESSED_DIR = os.getenv("PROCESSED_DIR", "../input/processed-data-msmarco/")
CORPUS_DIR = PROCESSED_DIR + os.getenv("CORPUS_DIR", "corpus_sampled/")
DATA_DIR = PROCESSED_DIR + os.getenv("CORPUS_DIR", "data_sampled/")
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "../input/latest-epoch.ckpt")
