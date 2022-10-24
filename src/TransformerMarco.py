import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers.optimization import get_constant_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import RetrievalNormalizedDCG
from typing import Tuple

from .msmarco import MarcoDataset, MarcoDataset2022
from .specs import ArgParams


class TransformerMarco(pl.LightningModule):
    """
    The Model. Impelements a few functions needed by PytorchLightning to do it's magic
    Important parts:
       __init__: initialize the model and all of it's parts
       forward: normal forward of the network
       configure_optimizers: configure optimizers
    """

    def __init__(self, hparams: ArgParams):
        # super().__init__()
        super(TransformerMarco, self).__init__()
        self.save_hyperparameters(hparams)
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            hparams.model_name
        )

        self.train_dataloader_object = (
            self.val_dataloader_object
        ) = self.test_dataloader_object = None
        self.DatasetClass = MarcoDataset

    def forward(self, input_ids, attention_mask, token_type_ids):

        outputs = self.model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        logits = outputs[0]

        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                factor=0.45,
            ),
            # "frequency": 900,
            # "interval": "step",
            # "monitor": "train_loss",
            "interval": "epoch",
            "monitor": "val_epoch_loss",
            "name": "reduce_lr_on_plateau",
        }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        dataset = self.DatasetClass(
            data_dir=self.hparams.data_dir,
            mode="train",
            tokenizer=self.tokenizer,
            max_seq_len=self.hparams.max_seq_len,
            args=self.hparams,
        )

        sampler = None
        # sampler = torch.utils.data.distributed.DistributedSampler(dataset) if self.trainer.use_ddp else None
        self.train_dataloader_object = DataLoader(
            dataset,
            batch_size=self.hparams.data_loader_bs,
            shuffle=(sampler is None),
            num_workers=self.hparams.num_workers,
            sampler=sampler,
            collate_fn=TransformerMarco.collate_fn,
        )
        return self.train_dataloader_object

    def val_dataloader(self):
        dataset = self.DatasetClass(
            data_dir=self.hparams.data_dir,
            mode="dev",
            tokenizer=self.tokenizer,
            max_seq_len=self.hparams.max_seq_len,
            args=self.hparams,
        )

        sampler = None
        # sampler = torch.utils.data.distributed.DistributedSampler(dataset) if self.trainer.use_ddp else None
        self.val_dataloader_object = DataLoader(
            dataset,
            batch_size=self.hparams.val_data_loader_bs,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            sampler=sampler,
            collate_fn=TransformerMarco.collate_fn,
        )
        return self.val_dataloader_object

    def test_dataloader(self):
        dataset = self.DatasetClass(
            data_dir=self.hparams.data_dir,
            mode="test",
            tokenizer=self.tokenizer,
            max_seq_len=self.hparams.max_seq_len,
            args=self.hparams,
        )

        sampler = None
        self.test_dataloader_object = DataLoader(
            dataset,
            batch_size=self.hparams.val_data_loader_bs,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            sampler=sampler,
            collate_fn=TransformerMarco.collate_fn,
        )
        return self.test_dataloader_object

    def training_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, labels, idxs = batch
        output = self.forward(input_ids, attention_mask, token_type_ids)

        loss = F.cross_entropy(output, labels.squeeze(1))
        if self.logger:
            self.logger.log_metrics({"train_loss": loss.item()})
        
        self.log_dict({"train_loss": loss.item()})

        # return {'loss': loss}
        return {"out": output, "labels": labels}

    def training_step_end(self, outputs):
        out = outputs["out"]
        labels = outputs["labels"].squeeze(1)
        loss = F.cross_entropy(out, labels)
        if self.logger:
            self.logger.log_metrics({"train_loss": loss.item()})
        
        self.log_dict({"train_loss": loss.item()})

        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, labels, idxs = batch
        output = self.forward(input_ids, attention_mask, token_type_ids)

        loss = F.cross_entropy(output, labels.squeeze(1))
        if self.logger:
            self.logger.log_metrics({"val_loss": loss.item()})
        
        self.log_dict({"val_loss": loss.item()})

        return {"out": output, "idxs": idxs, "labels": labels}
        # return {'loss': loss, 'probs': F.softmax(output, dim=1)[:,1], 'idxs': idxs}

    def validation_step_end(self, outputs):
        """
        outputs: dict of outputs of all batches in `dp` or `ddp2` mode
        """
        out = outputs["out"]
        labels = outputs["labels"]
        idxs = outputs["idxs"]

        loss = F.cross_entropy(out, labels.squeeze(1))
        if self.logger:
            self.logger.log_metrics({"val_loss": loss.item()})
        
        self.log_dict({"val_loss": loss.item()})

        return {"loss": loss, "probs": F.softmax(out, dim=1)[:, 1], "idxs": idxs}

    def validation_epoch_end(self, outputs):
        """
        outputs: dict of outputs of validation_step (or validation_step_end in dp/ddp2)
        outputs['loss'] --> losses of all the batches
        outputs['probs'] --> scores for each example
        outputs['idxs'] --> indexes in Dataset to connect with scores
        """

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        mrr, ndcg, rmap = self._get_retrieval_score(outputs)
        mrr10, ndcg10, rmap10 = self._get_retrieval_score(outputs, k=10)
        
        metric_dict = {
            "val_epoch_loss": avg_loss, 
            "mrr": mrr, 
            "mrr10": mrr10, 
            "ndcg": ndcg,
            "ndcg10": ndcg10,
            "map": rmap,
            "map10": rmap10,
        }

        if self.logger:
            self.logger.log_metrics(metric_dict)

        self.log_dict(metric_dict)

        print(f"\nDEV:: avg-LOSS: {avg_loss} || MRR: {mrr} || MRR@10: {mrr10} || NDCG: {ndcg} || NDCG@10: {ndcg10} || MAP: {rmap} || MAP@10: {rmap10}")
        
        metric_dict["progress_bar"] = metric_dict

        return metric_dict

    def _get_retrieval_score(self, outputs, k=None, mode="dev") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] :
        """Calculates MRR@k (Mean Reciprocal Rank)."""
        if mode == "dev":
            ds = self.val_dataloader_object.dataset
        elif mode == "test":
            ds = self.test_dataloader_object.dataset

        probs, idxs = [], []
        qids, dids, labels = [], [], []
        for x in outputs:
            probs += x["probs"].tolist()
            idxs += x["idxs"].tolist()

            top100_qids = ds.top100.iloc[x["idxs"].cpu()].qid.values.tolist()
            top100_dids = ds.top100.iloc[x["idxs"].cpu()].did.values.tolist()
            top100_labels = ds.top100.iloc[x["idxs"].cpu()].label.values.tolist()
            qids.extend(top100_qids)
            dids.extend(top100_dids)
            labels.extend(top100_labels)

        df = pd.DataFrame(
            {"prob": probs, "idx": idxs, "qid": qids, "did": dids, "label": labels}
        )
        
        mrr = 0.0
        rmap = 0.0
        for qid in df.qid.unique():
            tmp: pd.DataFrame = (
                df[df["qid"] == qid].sort_values("prob", ascending=False).reset_index()
            )
            if k:
                tmp = tmp.head(k)
            trues = tmp.index[tmp["label"] == 1].tolist()
            targets = tmp["label"].tolist()
            # if there is no relevant docs for this query
            if not trues:
                # add to total number of qids or not?
                pass
            else:
                first_relevant = trues[0] + 1  # pandas zero-indexing
                mrr += 1.0 / first_relevant

                prec = lambda x: sum(x) / len(x)
                sum_all = sum(targets)
                for i in range(len(targets)):
                    rmap += (prec(targets[:i + 1]) / sum_all) * targets[i]

        nqid = df.qid.nunique()
        mrr /= nqid
        rmap /= nqid

        probs = torch.tensor(probs)
        labels = torch.tensor(labels)
        qids = torch.tensor(qids)

        ndcg = RetrievalNormalizedDCG(k=k)

        mrr_score = torch.tensor(mrr)
        rmap_score = torch.tensor(rmap)
        ndcg_score = ndcg(probs, labels, indexes=qids)

        return mrr_score, ndcg_score, rmap_score

    @staticmethod
    def collate_fn(batch):
        input_ids = torch.stack([x["input_ids"] for x in batch])
        token_type_ids = torch.stack([torch.tensor(x["token_type_ids"]) for x in batch])
        attention_mask = torch.stack([x["attention_mask"] for x in batch])
        label = torch.stack([x["label"] for x in batch])
        idx = torch.stack([x["idx"] for x in batch])

        return (input_ids, attention_mask, token_type_ids, label, idx)

    def test_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, labels, idxs = batch
        output = self.forward(input_ids, attention_mask, token_type_ids)

        return {"probs": F.softmax(output, dim=1)[:, 1], "idxs": idxs}

    def test_epoch_end(self, outputs):
        """
        outputs: dict of outputs of test_step (or test_step_end in dp/ddp2)
        outputs['loss'] --> losses of all the batches
        outputs['probs'] --> scores for each example
        outputs['idxs'] --> indexes in Dataset to connect with scores
        """

        self._store_trec_output(outputs)
        return

    def _store_trec_output(self, outputs):

        ds = self.test_dataloader_object.dataset

        probs, idxs = [], []
        qids, dids = [], []
        for x in outputs:
            probs += x["probs"].tolist()
            idxs += x["idxs"].tolist()

            top1000_qids = ds.top100.iloc[x["idxs"].cpu()].qid.values.tolist()
            top1000_dids = ds.top100.iloc[x["idxs"].cpu()].did.values.tolist()
            for qid, did in zip(top1000_qids, top1000_dids):
                qids.append(qid)
                dids.append(did)

        df = pd.DataFrame({"prob": probs, "idx": idxs, "qid": qids, "did": dids})
        df["Q0"] = "Q0"
        df["run_name"] = self.hparams.run_name
        df["rank"] = df.groupby("qid")["prob"].rank(ascending=False)
        df.rank = df.rank.astype(int)
        df = df[["qid", "Q0", "did", "rank", "prob", "run_name"]]
        df.to_csv(f"{self.hparams.run_name}.tsv", sep=" ", header=False, index=False)
        return
