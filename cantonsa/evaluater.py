# coding=utf-8
import logging
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import OrderedDict 
import sklearn
from torch.utils.data import (
    DataLoader,
    SequentialSampler,
)
from tqdm import tqdm, trange
from cantonsa.constants import SENTI_ID_MAP_INV

logger = logging.getLogger(__name__)


class Evaluater():
    """
    
    """
    def __init__(
        self,
        model, 
        eval_config,
        output_dir,
        dataset,
        save_preds=False,
        save_reps=False,
        return_losses=False, 
        device=0,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.eval_config = eval_config
        self.output_dir = output_dir
        self.model = model
        self.dataset = dataset
        self.save_preds = save_preds
        self.save_reps = save_reps
        self.return_losses = return_losses
        self.scores = OrderedDict()

    def evaluation_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            inputs = dict()
            for col in self.model.INPUT_COLS:
                inputs[col] = batch[col].to(self.device)
            outputs = self.model(**inputs, return_reps=self.save_reps)
            losses = outputs[0].detach().cpu().numpy()
            predictions = torch.argmax(outputs[1], dim=1).detach().cpu().numpy()
            if self.save_reps:
                reps = outputs[2].detach().cpu().numpy()
            else:
                reps = None

        return losses, predictions, reps

    def save_scores(self):
        if len(self.scores) > 0:
            scores_df = pd.DataFrame.from_dict(self.scores, orient='index')
            scores_df.index.rename("identifier", inplace = True)
            scores_df.to_csv(self.output_dir / f"scores_{self.dataset.name}.csv")  

    @staticmethod
    def compute_metrics(labels, predictions, losses):
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
            )
            # "ars": ars(preds, labels, docids) if docids is not None else None,
        }
        metrics["avg_loss"] = np.mean(losses)
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
        )

        # Eval!
        logger.info("***** Running evaluation on %s*****", self.dataset.name)
        logger.info("  Num examples = %d", len(self.dataset))
        logger.info("  Batch size = %d", self.eval_config["batch_size"])

        # t0 = time.time()

        labels = []
        predictions = []
        losses = []
        representations = []
        
        for batch in tqdm(dataloader, desc="Evaluating"):

            outputs = self.evaluation_step(batch)
            losses.append(outputs[0])
            predictions.append(outputs[1])
            labels.append(batch["label"].detach().cpu().numpy())
            representations.append(outputs[2])

        labels = np.concatenate(labels, axis=None)
        predictions = np.concatenate(predictions, axis=None)
        losses = np.concatenate(losses, axis=None)
        representations = np.concatenate(representations, axis=0)

        metrics = self.compute_metrics(labels, predictions, losses)
        self.scores[identifier] = metrics

        if self.save_preds:
            df = self.dataset.get_df()
            df["prediction"] = predictions
            df["prediction"] = df["prediction"].map(SENTI_ID_MAP_INV)
            df["loss"] = losses
            out_path = self.output_dir / (f"preds_{self.dataset.name}_{identifier}.csv" if identifier else f"preds_{self.dataset.name}.csv")
            df.to_csv(
                out_path, 
                index=True, 
                encoding="utf-8-sig"
            )

        if self.save_reps:
            out_path = self.output_dir / (f"reps_{self.dataset.name}_{identifier}.npy" if identifier else f"reps_{self.dataset.name}.npy")
            np.save(out_path, representations)

        logger.info("***** Eval results *****")
        for key in sorted(metrics.keys()):
            logger.info("  {} = {:.4f}".format(key, metrics[key]))

        if self.return_losses:
            return (metrics, losses)
        else:
            return (metrics, )