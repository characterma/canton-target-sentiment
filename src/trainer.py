# coding=utf-8
import logging
import os
import sys
import time
from collections import defaultdict
import numpy as np
import pandas as pd
import copy
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


logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(
        self,
        model,
        train_config,
        optim_config,
        output_dir,
        dataset,
        dev_evaluaters,
        device=0,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.train_config = train_config
        self.optim_config = optim_config
        self.output_dir = output_dir
        self.model = model

        self.dataset = dataset
        self.dev_evaluaters = dev_evaluaters

        self.best_scores = defaultdict(lambda: defaultdict(lambda: None))
        self.best_epoch = defaultdict(lambda: None)
        self.best_step = defaultdict(lambda: None)
        self.best_model = defaultdict(lambda: None)

    def _get_train_sampler(self):

        # create sampler for training
        if (
            "sampler" not in self.optim_config
            or self.optim_config["sampler"].lower() == "random"
        ):
            train_sampler = RandomSampler(self.dataset)
        elif self.optim_config["sampler"].lower() == "ros":
            (
                class_balanced_weights,
                majority_size,
                _,
                num_cls,
            ) = self.dataset.get_class_balanced_weights()
            num_samples = int(majority_size * num_cls)
            train_sampler = WeightedRandomSampler(
                weights=class_balanced_weights,
                num_samples=num_samples,
                replacement=True,
            )
        elif self.optim_config["sampler"].lower() == "rus":
            (
                class_balanced_weights,
                majority_size,
                minority_size,
                num_cls,
            ) = self.dataset.get_class_balanced_weights()
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
        if self.optim_config["max_steps"] > 0:
            t_total = self.optim_config["max_steps"]
            self.optim_config["num_train_epochs"] = (
                self.optim_config["max_steps"]
                // (len(dataloader) // self.optim_config["gradient_accumulation_steps"])
                + 1
            )
        else:
            t_total = (
                len(dataloader)
                // self.optim_config["gradient_accumulation_steps"]
                * self.optim_config["num_train_epochs"]
            )
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": float(self.optim_config["weight_decay"]),
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
        if self.optim_config["optimizer"] == "AdamW":
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=float(self.optim_config["learning_rate"]),
                eps=float(self.optim_config["adam_epsilon"]),
            )
        elif self.optim_config["optimizer"] == "RMSprop":
            optimizer = RMSprop(
                optimizer_grouped_parameters,
                lr=float(self.optim_config["learning_rate"]),
                eps=float(self.optim_config["adam_epsilon"]),
            )
        else:
            raise (Exception)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.optim_config["warmup_steps"]),
            num_training_steps=t_total,
        )
        return optimizer, scheduler

    def training_step(self, batch, step):
        # step 0: only sentiment classification
        # step 1: sentiment classification + token classification
        self.model.train()
        inputs = dict()
        # only sentiment classification
        for col in self.model.INPUT_COLS:
            if col in batch:
                if col!="soft_label":
                    inputs[col] = batch[col].to(self.device).long()
                else:
                    inputs[col] = batch[col].to(self.device).float()
        outputs = self.model(**inputs)
        losses = outputs[0]
        logits = outputs[1]
        losses.mean().backward()
        return losses.detach()

    def train(self):

        # freeze backbone parameters
        if "freeze_lm" in self.optim_config and self.optim_config["freeze_lm"]:
            self.model.freeze_lm()

        # reset classifier parameters
        if (
            "reset_classifier" in self.optim_config
            and self.optim_config["reset_classifier"]
        ):
            self.model.init_classifier()

        dataloader = DataLoader(
            self.dataset,
            shuffle=True, 
            batch_size=self.optim_config["batch_size"],
            collate_fn=self.dataset.pad_collate,
        )

        optimizer, scheduler = self.create_optimizer_and_scheduler(dataloader)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.dataset))
        logger.info("  Num Epochs = %d", self.optim_config["num_train_epochs"])
        logger.info("  Sampler = %s", self.optim_config.get("sampler", ""))
        logger.info("  Batch size = %d", self.optim_config["batch_size"])
        logger.info(
            "  Gradient Accumulation steps = %d",
            self.optim_config["gradient_accumulation_steps"],
        )
        logger.info("  Logging steps = %d", self.train_config["logging_steps"])

        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(
            int(self.optim_config["num_train_epochs"]),
            desc=f"Epoch",
        )

        for epoch, _ in enumerate(train_iterator):

            epoch_iterator = tqdm(dataloader, desc="Iteration")
            last_step = len(epoch_iterator)
            for step, batch in enumerate(epoch_iterator):

                losses = self.training_step(batch, step)
                losses = losses.cpu().numpy()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                epoch_iterator.set_postfix({"tr_loss": np.mean(losses)})

                if (step + 1) % self.optim_config["gradient_accumulation_steps"] == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.optim_config["max_grad_norm"],
                    )

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()

            self.on_epoch_end(epoch)
        self.on_training_end()

    def on_epoch_end(self, epoch):

        for evaluater in self.dev_evaluaters:
            outputs = evaluater.evaluate(identifier=f"epoch{epoch}")
            metrics = outputs[0]

            # determine whether it is best
            # if evaluater.dataset_name!="train":
            if (
                self.best_scores[evaluater.dataset_name]["macro_f1"] is None
                or self.best_scores[evaluater.dataset_name]["macro_f1"]
                < metrics["macro_f1"]
            ):
                self.best_scores[evaluater.dataset_name] = metrics
                self.best_epoch[evaluater.dataset_name] = epoch
                self.best_model[evaluater.dataset_name] = copy.deepcopy(
                    self.model.state_dict()
                )

    def on_training_end(self):
        """
        1. Save best model states.
        2. Save evaluation scores.
        """
        for dataset_name in self.best_model.keys():
            if self.best_model[dataset_name] is not None:
                epoch = self.best_epoch[dataset_name]
                out_path = (
                    self.output_dir / f"best_state_{dataset_name}_epoch{epoch}.pt"
                )
                torch.save(self.best_model[dataset_name], out_path)
