import os
import pandas as pd
import torch
import json
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from .constants import DATA_DIR, DOCS_FILE, DOCS_DIR, INPUT_DIR, TEST_DIR
from .specs import ArgParams


class MarcoDataset2022(Dataset):
    """
    Dataset abstraction for MS MARCO document re-ranking.
    """
    data_dir = DATA_DIR
    docs_dir = DOCS_DIR
    def __init__(self, data_dir=None, mode="train", tokenizer=None, max_seq_len=512, args=None):
        self.data_dir = data_dir or self.data_dir
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        # load queries
        self.queries = pd.read_csv(
            os.path.join(self.data_dir, f"docv2_{mode}_queries.tsv"),
            sep="\t",
            header=None,
            names=["qid", "query_text"],
            index_col="qid",
        )
        self.relations = pd.read_csv(
            os.path.join(self.data_dir, f"docv2_{mode}_qrels.tsv"),
            sep="\t",
            header=None,
            names=["qid", "0", "did", "label"],
        )
        self.top100 = pd.read_csv(
            os.path.join(self.data_dir, f"docv2_{mode}_top100.tsv"),
            sep=" ",
            header=None,
            names=["qid", "Q0", "did", "rank", "score", "run"],
            dtype={'qid': 'int32', "rank": "int8", "score": "float16"},
        )

        # downsample the dataset so the positive:negative ratio is 1:10
        if mode == "train":
            self.top100 = self.top100.sample(frac=0.01, random_state=42).append(
                self.relations[["qid", "did"]], ignore_index=True
            )
            self.top100.drop_duplicates(keep="first", inplace=True)
            # shuffle the data so positives are ~ evenly distributed
            self.top100 = self.top100.sample(frac=1, random_state=42).reset_index(
                drop=True
            )

        elif mode == "dev" and args.use_10_percent_of_dev:
            # use 10% of the data for dev during training
            import numpy as np

            np.random.seed(42)
            queries = self.top100["qid"].unique()
            queries = np.random.choice(queries, int(len(queries) / 50), replace=False)
            print(len(queries))
            self.top100 = self.top100[self.top100["qid"].isin(queries)]

        print(f"{mode} set len:", len(self.top100))

    # needed for map-style torch Datasets
    def __len__(self):
        return len(self.top100)

    def _get_document(self, document_id):
        (string1, string2, bundlenum, position) = document_id.split('_')
        assert string1 == 'msmarco' and string2 == 'doc'

        with open(f'{self.docs_dir}/msmarco_doc_{bundlenum}', 'rt', encoding='utf8') as in_fh:
            in_fh.seek(int(position))
            json_string = in_fh.readline()
            document = json.loads(json_string)

            # dict_keys(['url', 'title', 'headings', 'body', 'docid'])
            assert document['docid'] == document_id

            return document

    # needed for map-style torch Datasets
    def __getitem__(self, idx):
        x = self.top100.iloc[idx]
        query = self.queries.loc[x.qid].query_text
        document = self._get_document(x.did)['body']

        label = (
            0
            if self.relations.loc[
                (self.relations["qid"] == x.qid) & (self.relations["did"] == x.did)
            ].empty
            else 1
        )

        tensors = self.one_example_to_tensors(query, document, idx, label)
        return tensors

    # main method for encoding the example
    def one_example_to_tensors(self, query, document, idx, label):

        encoded = self.tokenizer.encode_plus(
            query,
            document,
            add_special_tokens=True,
            max_length=self.max_seq_len,
            truncation="only_second",
            truncation_strategy="only_second",
            return_overflowing_tokens=False,
            return_special_tokens_mask=False,
            return_token_type_ids=True,
            pad_to_max_length=True,
        )
        encoded["attention_mask"] = torch.tensor(encoded["attention_mask"])

        encoded["input_ids"] = torch.tensor(encoded["input_ids"])

        encoded.update({"label": torch.LongTensor([label]), "idx": torch.tensor(idx)})
        return encoded


class MarcoDataset(Dataset):
    """
    Dataset abstraction for MS MARCO document re-ranking.
    """
    data_dir = DATA_DIR
    docs_file = DOCS_FILE
    def __init__(self, data_dir=None, mode="train", tokenizer=None, max_seq_len=512, args: ArgParams=None):
        self.data_dir = data_dir or self.data_dir
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # self.relations = pd.read_csv(
        #     os.path.join(INPUT_DIR, f"msmarco-doc{mode}-qrels.tsv"),
        #     sep=" ",
        #     header=None,
        #     names=["qid", "0", "did", "label"],
        # )
        if mode == "dev":
            self.top100 = pd.read_csv(
                os.path.join(DATA_DIR, f"validation_{args.validation}k.csv"),
                dtype={'qid': 'int32', 'rank': 'int8', "score": "float16"},
                usecols=["qid", "did", "label", "rank", "score"],
            )
        else:
            self.top100 = pd.read_csv(
                os.path.join(TEST_DIR, f"test_{args.test}.csv"),
                dtype={'qid': 'int32', 'rank': 'int8', "score": "float16"},
                usecols=["qid", "did", "label", "rank", "score"],
            )
            print(f"Size test {args.test}")

        if mode == "train":
            self.data = pd.read_csv(
                os.path.join(DATA_DIR, f"training_{args.training}k.csv"),
                dtype={'qid': 'int32', "rank": "int8", "score": "float16"},
                usecols=["qid", "did", "label", "query", "doc"],
            )
        elif mode == "dev":
            self.data = pd.read_csv(
                os.path.join(DATA_DIR, f"validation_{args.validation}k.csv"),
                dtype={'qid': 'int32'},
                usecols=["qid", "did", "label", "query", "doc"],
            )
        else:
            self.data = pd.read_csv(
                os.path.join(TEST_DIR, f"test_{args.test}.csv"),
                dtype={'qid': 'int32'},
                usecols=["qid", "did", "label", "query", "doc"],
            )
            print(f"Size test {args.test}")

        print(f"{mode} set len:", len(self.data))

    # needed for map-style torch Datasets
    def __len__(self):
        return len(self.data)

    # needed for map-style torch Datasets
    def __getitem__(self, idx):
        x = self.data.iloc[idx]
        tensors = self.one_example_to_tensors(x["query"], x["doc"], idx, x["label"])
        return tensors

    # main method for encoding the example
    def one_example_to_tensors(self, query, document, idx, label):

        encoded = self.tokenizer.encode_plus(
            query,
            document,
            add_special_tokens=True,
            max_length=self.max_seq_len,
            truncation="only_second",
            truncation_strategy="only_second",
            return_overflowing_tokens=False,
            return_special_tokens_mask=False,
            return_token_type_ids=True,
            padding="max_length",
        )
        encoded["attention_mask"] = torch.tensor(encoded["attention_mask"])

        encoded["input_ids"] = torch.tensor(encoded["input_ids"])

        encoded.update({"label": torch.LongTensor([label]), "idx": torch.tensor(idx)})
        return encoded
