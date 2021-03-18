# coding=utf-8
import logging
import time
import numpy as np
import pandas as pd
import torch
from collections import OrderedDict
import sklearn
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from tqdm import tqdm, trange
from utils import SENTI_ID_MAP_INV

logger = logging.getLogger(__name__)


class Evaluater:
    def __init__(
        self,
        model,
        eval_config,
        output_dir,
        dataset,
        timer=None,
        no_save=False,
        device=0,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.eval_config = eval_config
        self.output_dir = output_dir
        self.model = model
        self.dataset = dataset
        self.metrics = OrderedDict()
        self.no_save = no_save
        self.timer = timer

    def evaluation_step(self, batch):

        self.model.eval()
        with torch.no_grad():
            inputs = dict()
            for col in self.model.INPUT_COLS:
                if col != "soft_label":
                    inputs[col] = batch[col].to(self.device).long()
            results = self.model(
                **inputs,
            )
            outputs = dict()

            # alwasy output
            outputs["sentiment_idx"] = torch.argmax(results[1], dim=1)
            # print(outputs["logits"])
            outputs["loss"] = torch.mean(results[0])
            # print(outputs["loss"])
            outputs["logits"] = results[1]
            outputs["score"] = torch.nn.functional.softmax(results[1], dim=1)
        return outputs

    def save_metrics(self):
        if len(self.metrics) > 0:
            metrics_df = pd.DataFrame.from_dict(self.metrics, orient="index")
            metrics_df.index.rename("identifier", inplace=True)
            metrics_df.to_csv(self.output_dir / f"metrics_{self.dataset.name}.csv")

    @staticmethod
    def compute_metrics(labels, predictions):
        assert len(predictions) == len(labels)
        acc_repr = sklearn.metrics.classification_report(
            labels, predictions, labels=[0, 1, 2], output_dict=True
        )
        metrics = {
            "acc": (predictions == labels).mean(),
            "macro_f1": sklearn.metrics.f1_score(
                labels, predictions, labels=[0, 1, 2], average="macro"
            ),
            "micro_f1": sklearn.metrics.f1_score(
                labels, predictions, labels=[0, 1, 2], average="micro"
            ),
        }
        for senti_id, v1 in acc_repr.items():
            if not senti_id.isdigit():
                continue
            senti_name = SENTI_ID_MAP_INV.get(int(senti_id), None)
            if senti_name is not None:
                for score_name, v2 in v1.items():
                    metrics[f"{senti_name}-{score_name}"] = v2
        return metrics

    @property
    def dataset_name(self):
        return self.dataset.name

    def evaluate(self, identifier=""):

        dataloader = DataLoader(
            self.dataset,
            sampler=SequentialSampler(self.dataset),
            batch_size=self.eval_config["batch_size"],
            collate_fn=self.dataset.pad_collate,
        )

        # Eval!
        logger.info("***** Running evaluation on %s*****", self.dataset.name)
        logger.info("  Num examples = %d", len(self.dataset))
        logger.info("  Batch size = %d", self.eval_config["batch_size"])

        # basic info
        label = np.array([])
        sentiment_idx = np.array([])
        score = np.array([])
        logits = np.array([])
        losses = np.array([])

        if self.timer is not None:
            self.timer.on_inference_start()

        for batch in tqdm(dataloader, desc="Evaluating"):

            results = self.evaluation_step(batch)

            label = np.concatenate(
                [label, batch["label"].detach().cpu().numpy()], axis=None
            )
            losses = np.concatenate(
                [losses, results["loss"].detach().cpu().numpy()], axis=None
            )

            sentiment_idx = np.concatenate(
                [sentiment_idx, results["sentiment_idx"].detach().cpu().numpy()],
                axis=None,
            )
            score = np.concatenate(
                [score, results["score"].detach().cpu().numpy().max(axis=1)], axis=None
            )
            if len(logits)==0:
                logits = results["logits"].detach().cpu().numpy()
            else:
                logits = np.concatenate(
                    [logits, results["logits"].detach().cpu().numpy()], axis=0
                )

        if self.timer is not None:
            self.timer.on_inference_end()
        self.metrics[identifier] = self.compute_metrics(label, sentiment_idx)
        self.metrics[identifier]['loss'] = losses.mean()

        # save basic info
        if not self.no_save:
            np.save(self.output_dir / "logits.npy", logits)
            df = self.dataset.df.copy()
            df["label_idx"] = label
            df["sentiment_idx"] = sentiment_idx
            df["sentiment"] = df["sentiment_idx"].map(SENTI_ID_MAP_INV)
            df["score"] = score
            df.to_csv(
                self.output_dir / f"eval_details_{self.dataset.name}_{identifier}.csv",
                index=True,
                encoding="utf-8-sig",
            )

        # save eval metrics
        pd.DataFrame(data=[self.metrics[identifier]]).to_csv(
            self.output_dir / f"metrics_{self.dataset.name}_{identifier}.csv",
            index=False,
        )

        logger.info("***** Eval results *****")
        for key in sorted(self.metrics[identifier].keys()):
            logger.info("  {} = {:.4f}".format(key, self.metrics[identifier][key]))

        return (self.metrics[identifier],)
