# coding=utf-8
import logging
import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
import copy
import sklearn
import torch
import torch.nn as nn
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    WeightedRandomSampler,
)
from torch.optim import RMSprop
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import SENTI_ID_MAP_INV


logger = logging.getLogger(__name__)


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


def evaluation_step(model, batch, device):
    model.eval()
    results = dict()
    with torch.no_grad():
        inputs = dict()
        for col in model.INPUT:
            inputs[col] = batch[col].to(device).long()
        x = model(
            **inputs,
        )
    results["sentiment_idx"] = torch.argmax(x[1], dim=1)
    results["loss"] = torch.mean(x[0])
    results["logits"] = x[1]
    results["score"] = torch.nn.functional.softmax(x[1], dim=1)
    return results


def evaluate(model, eval_dataset, args):

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_config["batch_size"])

    dataloader = DataLoader(
        eval_dataset,
        shuffle=False, 
        batch_size=args.eval_config["batch_size"],
        # collate_fn=eval_dataset.pad_collate,
    )

    label = np.array([])
    sentiment_idx = np.array([])
    score = np.array([])
    logits = np.array([])
    losses = np.array([])

    for batch in tqdm(dataloader, desc="Evaluating"):

        results = evaluation_step(model, batch, device=args.device)

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

    metrics = compute_metrics(label, sentiment_idx)
    metrics['loss'] = losses.mean()
    return metrics


class Trainer(object):
    def __init__(
        self,
        model,
        train_dataset,
        dev_dataset,
        args
    ):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.model = model
        self.device = args.device if torch.cuda.is_available() else "cpu"

        self.model_config = args.model_config
        self.train_config = args.train_config
        self.out_dir = args.out_dir

        self.best_score = None
        self.best_epoch = None
        self.best_step = None
        self.best_model_state = None

    def _get_train_sampler(self):

        # create sampler for training
        if (
            "sampler" not in self.model_config
            or self.model_config["sampler"].lower() == "random"
        ):
            train_sampler = RandomSampler(self.train_dataset)
        elif self.model_config["sampler"].lower() == "ros":
            (
                class_balanced_weights,
                majority_size,
                _,
                num_cls,
            ) = self.train_dataset.get_class_balanced_weights()
            num_samples = int(majority_size * num_cls)
            train_sampler = WeightedRandomSampler(
                weights=class_balanced_weights,
                num_samples=num_samples,
                replacement=True,
            )
        elif self.model_config["sampler"].lower() == "rus":
            (
                class_balanced_weights,
                majority_size,
                minority_size,
                num_cls,
            ) = self.train_dataset.get_class_balanced_weights()
            num_samples = int(minority_size * num_cls)
            train_sampler = WeightedRandomSampler(
                weights=class_balanced_weights,
                num_samples=num_samples,
                replacement=False,
            )
        else:
            raise (Exception)
        return train_sampler

    def create_optimizer_and_scheduler(self, dataloader):
        if self.model_config["max_steps"] > 0:
            t_total = self.model_config["max_steps"]
            self.model_config["num_train_epochs"] = (
                self.model_config["max_steps"]
                // (len(dataloader) // self.model_config["gradient_accumulation_steps"])
                + 1
            )
        else:
            t_total = (
                len(dataloader)
                // self.model_config["gradient_accumulation_steps"]
                * self.model_config["num_train_epochs"]
            )
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": float(self.model_config["weight_decay"]),
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        if self.model_config["optimizer"] == "AdamW":
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=float(self.model_config["learning_rate"]),
                eps=float(self.model_config["adam_epsilon"]),
            )
        elif self.model_config["optimizer"] == "RMSprop":
            optimizer = RMSprop(
                optimizer_grouped_parameters,
                lr=float(self.model_config["learning_rate"]),
                eps=float(self.model_config["adam_epsilon"]),
            )
        else:
            raise (Exception)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.model_config["warmup_steps"]),
            num_training_steps=t_total,
        )
        return optimizer, scheduler

    def training_step(self, batch):
        self.model.train()
        inputs = dict()
        for col in self.model.INPUT:
            inputs[col] = batch[col].to(self.device).long()
        outputs = self.model(**inputs)
        losses = outputs[0]
        logits = outputs[1]
        losses.mean().backward()
        return losses.detach()

    def train(self):

        dataloader = DataLoader(
            self.train_dataset,
            sampler=self._get_train_sampler(), 
            batch_size=self.model_config["batch_size"],
            # collate_fn=self.train_dataset.pad_collate,
        )

        optimizer, scheduler = self.create_optimizer_and_scheduler(dataloader)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.model_config["num_train_epochs"])
        logger.info("  Sampler = %s", self.model_config.get("sampler", ""))
        logger.info("  Batch size = %d", self.train_config["batch_size"])
        logger.info(
            "  Gradient Accumulation steps = %d",
            self.model_config["gradient_accumulation_steps"],
        )
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(
            int(self.model_config["num_train_epochs"]),
            desc=f"Epoch",
        )
        for epoch, _ in enumerate(train_iterator):
            epoch_iterator = tqdm(dataloader, desc="Iteration")
            last_step = len(epoch_iterator)
            for step, batch in enumerate(epoch_iterator):

                losses = self.training_step(batch)
                losses = losses.cpu().numpy()

                epoch_iterator.set_postfix({"tr_loss": np.mean(losses)})

                if (step + 1) % self.model_config["gradient_accumulation_steps"] == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.model_config["max_grad_norm"],
                    )

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()

            self.on_epoch_end(epoch)
        self.on_training_end()

    def on_epoch_end(self, epoch):

        metrics = evaluate(
            model=self.model,
            eval_dataset=self.dev_dataset,
            args=self.args,
        )

        if self.best_score is None or self.best_score < metrics["loss"]:
            self.best_score = metrics["loss"]
            self.best_model_state = copy.deepcopy(self.model.state_dict())

    def on_training_end(self):
        """
        # 1. Save best model states.
        # 2. Save evaluation scores.
        """
        out_path = (
            self.out_dir / f"best_state.pt"
        )
        torch.save(self.best_model_state, out_path)

        self.model.load_state(
            state_dict=self.best_model_state
        )
