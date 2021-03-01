# coding=utf-8
import logging
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import OrderedDict
import sklearn
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from tqdm import tqdm, trange
from utils import SENTI_ID_MAP_INV
import pickle

logger = logging.getLogger(__name__)


class Evaluater:
    def __init__(
        self,
        model,
        eval_config,
        output_dir,
        dataset,
        return_loss=False,
        timer=None,
        no_save=False,
        device=0,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.eval_config = eval_config
        self.output_dir = output_dir
        self.model = model
        self.dataset = dataset
        self.return_loss = return_loss
        self.metrics = OrderedDict()
        self.no_save = no_save
        self.timer = timer

    def evaluation_step(self, batch):

        print("###########", batch.keys())

        self.model.eval()
        with torch.no_grad():
            inputs = dict()
            for col in self.model.INPUT_COLS:
                inputs[col] = batch[col].to(self.device)
            results = self.model(
                **inputs,
                return_tgt_pool=self.eval_config["save_tgt_pool"] and not self.no_save,
                return_tgt_mask=self.eval_config["save_tgt_mask"] and not self.no_save,
                return_all_repr=self.eval_config["save_all_repr"] and not self.no_save,
                return_attn=self.eval_config["save_attn"] and not self.no_save,
            )
            outputs = dict()

            # alwasy output
            outputs["loss"] = results[0]
            outputs["sentiment_idx"] = torch.argmax(results[1], dim=1)
            outputs["score"] = torch.nn.functional.softmax(results[1], dim=1)

            # output when needed, otherwise None
            outputs["tgt_pool"] = results[2]
            outputs["tgt_mask"] = results[3]
            outputs["all_repr"] = results[4]
            outputs["attn"] = results[5]
        return outputs

    def save_metrics(self):
        if len(self.metrics) > 0:
            metrics_df = pd.DataFrame.from_dict(self.metrics, orient="index")
            metrics_df.index.rename("identifier", inplace=True)
            metrics_df.to_csv(self.output_dir / f"metrics_{self.dataset.name}.csv")

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

        # basic info
        label = np.array([])
        sentiment_idx= np.array([])
        loss = np.array([])
        score = np.array([])

        # additional info
        tgt_pool = None
        tgt_mask= None
        all_repr = None
        attn = None        

        if self.timer is not None:
            self.timer.on_inference_start()

        for batch in tqdm(dataloader, desc="Evaluating"):

            results = self.evaluation_step(batch)

            # concat batch results
            label = np.concatenate([label, batch["label"].detach().cpu().numpy()], axis=None)
            sentiment_idx = np.concatenate([sentiment_idx, results["sentiment_idx"].detach().cpu().numpy()], axis=None)
            loss = np.concatenate([loss, results["loss"].detach().cpu().numpy()], axis=None)
            score = np.concatenate([score, results["score"].detach().cpu().numpy().max(axis=1)], axis=None)
            
            if results["tgt_pool"] is not None:
                if tgt_pool is None:
                    tgt_pool = results["tgt_pool"].detach().cpu().numpy()
                else:
                    tgt_pool = np.concatenate([tgt_pool, results["tgt_pool"].detach().cpu().numpy()], axis=0)
            if results["tgt_mask"] is not None:
                if tgt_mask is None:
                    tgt_mask = results["tgt_mask"].detach().cpu().numpy()
                else:
                    tgt_mask = np.concatenate([tgt_mask, results["tgt_mask"].detach().cpu().numpy()], axis=0)
            if results["all_repr"] is not None:
                if all_repr is None:
                    all_repr = results["all_repr"].detach().cpu().numpy()
                else:
                    all_repr = np.concatenate([all_repr, results["all_repr"].detach().cpu().numpy()], axis=0)
            if results["attn"] is not None:
                _attn = np.array([a.detach().cpu().numpy() for a in results["attn"]])
                if attn is None:
                    attn = [_attn]
                else:
                    attn.append(_attn)

        if self.timer is not None:
            self.timer.on_inference_end()

        self.metrics[identifier] = self.compute_metrics(label, sentiment_idx, loss)

        # save basic info

        if not self.no_save:
            df = self.dataset.df.copy()
            df["label_idx"] = label
            df["sentiment_idx"] = sentiment_idx
            df["sentiment"] = df["sentiment_idx"].map(SENTI_ID_MAP_INV)
            df["score"] = score
            df["loss"] = loss
            df.to_csv(self.output_dir / f"eval_details_{self.dataset.name}_{identifier}.csv", index=True, encoding="utf-8-sig")

        # save additional info
        if tgt_pool is not None:
            np.save(self.output_dir / f"tgt_pool_{self.dataset.name}_{identifier}.npy", tgt_pool)
        if tgt_mask is not None:
            np.save(self.output_dir / f"tgt_mask_{self.dataset.name}_{identifier}.npy", tgt_mask)
        if all_repr is not None:
            np.save(self.output_dir / f"all_repr_{self.dataset.name}_{identifier}.npy", all_repr)
        if attn is not None:
            attn = np.concatenate(attn, axis=1)
            np.save(self.output_dir / f"attn_{self.dataset.name}_{identifier}.npy", attn)

            # with open(self.output_dir / f"attn_{self.dataset.name}_{identifier}.pkl", 'wb') as f:
            #     pickle.dump(attn, f)

        # save eval metrics
        pd.DataFrame(data=[self.metrics[identifier]]).to_csv(
            self.output_dir / f"metrics_{self.dataset.name}_{identifier}.csv", 
            index=False
        )

        logger.info("***** Eval results *****")
        for key in sorted(self.metrics[identifier].keys()):
            logger.info("  {} = {:.4f}".format(key, self.metrics[identifier][key]))

        if self.return_loss:
            return (self.metrics[identifier], loss)
        else:
            return (self.metrics[identifier],)
