import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup

from model import SAModel
from utils import compute_metrics, get_label_map, write_eval_details

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, train_config, model_config, data_config, pretrained_model, label_map, model_dir, train_dataset=None, dev_dataset=None, test_dataset=None, 
                train_details=None, dev_details=None, test_details=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train_config = train_config
        self.data_config = data_config
        self.model_config = model_config
        self.model_dir = model_dir

        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.pretrained_model = pretrained_model
        
        self.train_details = train_details
        self.dev_details = dev_details
        self.test_details = test_details

        self.label_map = label_map
        self.num_labels = len(self.label_map.values())

        self.model = SAModel(model_config, self.num_labels, self.pretrained_model, device=self.device)

        # GPU or CPU
        # self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=self.train_config["optim"]["train_batch_size"],
        )

        if self.train_config["optim"]["max_steps"] > 0:
            t_total = self.train_config["optim"]["max_steps"]
            self.train_config["optim"]["num_train_epochs"] = (
                self.train_config["optim"]["max_steps"]
                // (len(train_dataloader) // self.train_config["optim"]["gradient_accumulation_steps"])
                + 1
            )
        else:
            t_total = (
                len(train_dataloader)
                // self.train_config["optim"]["gradient_accumulation_steps"]
                * self.train_config["optim"]["num_train_epochs"]
            )

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.train_config["optim"]["weight_decay"],
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
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.train_config["optim"]["learning_rate"],
            eps=self.train_config["optim"]["adam_epsilon"],
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.train_config["optim"]["warmup_steps"],
            num_training_steps=t_total,
        )

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.train_config["optim"]["num_train_epochs"])
        logger.info("  Total train batch size = %d", self.train_config["optim"]["train_batch_size"])
        logger.info(
            "  Gradient Accumulation steps = %d", self.train_config["optim"]["gradient_accumulation_steps"]
        )
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.train_config["log"]["logging_steps"])

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.train_config["optim"]["num_train_epochs"]), desc="Epoch")
        best_eval_score = None
        all_eval_scores = []
        for epoch, _ in enumerate(train_iterator):
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                inputs = {
                    "input_ids": batch[1],
                    "attention_mask": batch[2],
                    "token_type_ids": batch[3],
                    "labels": batch[4],
                    "t_mask": batch[5],
                }
                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.train_config["optim"]["gradient_accumulation_steps"] > 1:
                    loss = loss / self.train_config["optim"]["gradient_accumulation_steps"]

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.train_config["optim"]["gradient_accumulation_steps"] == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.train_config["optim"]["max_grad_norm"]
                    )

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if (
                        self.train_config["log"]["logging_steps"] > 0
                        and global_step % self.train_config["log"]["logging_steps"] == 0
                    ):
                        eval_scores = self.evaluate("dev", f"epoch{epoch}_step{step}")
                        eval_scores["epoch"] = epoch
                        eval_scores["step"] = step
                        all_eval_scores.append(all_eval_scores)

                        if best_eval_score is None:
                            best_eval_score = eval_scores['loss']
                        elif best_eval_score > eval_scores['loss']:
                            best_eval_score = eval_scores['loss']
                            self.model.save_state(self.model_dir, suffix=f"epoch{epoch}_step{step}")

                if 0 < self.train_config["optim"]["max_steps"] < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.train_config["optim"]["max_steps"] < global_step:
                train_iterator.close()
                break
        pd.DataFrame(data = all_eval_scores).to_csv(self.model_dir / "eval_scores.csv")
        return global_step, tr_loss / global_step

    def evaluate(self, mode, suffix=""):
        # We use test dataset because semeval doesn't have dev dataset
        if mode == "test":
            dataset = self.test_dataset
            details = self.test_details
        elif mode == "dev":
            dataset = self.dev_dataset
            details = self.dev_details
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(
            dataset, sampler=eval_sampler, batch_size=self.train_config["optim"]["eval_batch_size"]
        )

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.train_config["optim"]["eval_batch_size"])
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        keys = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[1],
                    "attention_mask": batch[2],
                    "token_type_ids": batch[3],
                    "labels": batch[4],
                    "t_mask": batch[5],
                }
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
                keys = batch[0].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
                )
                keys = np.append(
                    keys, batch[0].detach().cpu().numpy(), axis=0
                )
        if suffix:
            eval_details_path = self.model_dir / f"eval_details_{mode}_{suffix}.csv"
        else:
            eval_details_path = self.model_dir / f"eval_details_{mode}.csv"
        

        eval_loss = eval_loss / nb_eval_steps
        results = {"loss": eval_loss}
        preds = np.argmax(preds, axis=1)
        result = compute_metrics(preds, out_label_ids)
        results.update(result)
        write_eval_details(self.data_config, eval_details_path, details, preds, keys)


        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  {} = {:.4f}".format(key, results[key]))

        return results

