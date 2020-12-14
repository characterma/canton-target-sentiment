import logging
import os
import sys
import time
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    WeightedRandomSampler,
)
from torch.optim import RMSprop
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup

from cantonsa.models import *
from cantonsa.utils import compute_metrics, get_label_map, join_eval_details
from cantonsa.constants import MODEL_EMB_TYPE


logger = logging.getLogger(__name__)


class Pipeline(object):
    def __init__(
        self,
        train_id,
        train_config,
        eval_config,
        body_config,
        optim_config,
        data_config,
        label_map,
        output_dir,
        pretrained_lm=None,
        pretrained_word_emb=None,
        train_dataset=None,
        dev_dataset=None,
        test_dataset=None,
        state_path=None,
        save_eval_scores=True,
        save_eval_details=True,
        global_best_scores=None,
        num_emb=None,
        device=0,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.train_id = train_id
        self.train_config = train_config
        self.eval_config = eval_config
        self.data_config = data_config
        self.body_config = body_config
        self.optim_config = optim_config
        self.output_dir = output_dir
        self.state_path = state_path

        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.num_emb = num_emb

        # self.vocab_size = vocab_size # used for TGSAN
        self.pretrained_lm = pretrained_lm
        self.pretrained_emb = (
            pretrained_word_emb.vectors if pretrained_word_emb is not None else None
        )

        self.label_map = label_map
        self.num_labels = len(self.label_map.values())
        self.global_best_scores = global_best_scores

        self.input_col_dict = {
            "TDBERT": [
                "raw_text",
                "attention_mask",
                "token_type_ids",
                "target_mask",
                "label",
            ],
            "TGSAN": ["raw_text", "attention_mask", "target_mask", "label"],
            "TDLSTM": ["target_left_inclu", "target_right_inclu", "label"],
            "TNET_LF": ["raw_text", "target", "target_span", "label"],
            "RAM": ["raw_text", "target", "target_left", "label"],
            "MEMNET": ["raw_text_without_target", "target", "label"],
            "IAN": ["raw_text", "target", "label"],
            "ATAE_LSTM": ["raw_text", "target", "label"],
        }
        self.save_eval_scores = save_eval_scores
        self.save_eval_details = save_eval_details

        self.model_class = train_config["model_class"]
        logger.info(f"***** Model Class : {self.model_class} *****")

        MODEL = getattr(sys.modules[__name__], self.model_class)
        self.model = MODEL(
            model_config=body_config,
            num_labels=self.num_labels,
            pretrained_emb=self.pretrained_emb,
            num_emb=self.num_emb,
            pretrained_lm=self.pretrained_lm,
            device=self.device,
        )
        if self.state_path is not None:
            self.model.load_state(self.state_path)

    def train(self):
        best_eval_scores = defaultdict(lambda: None)
        best_eval_epoch = defaultdict(lambda: None)
        best_eval_step = defaultdict(lambda: None)
        all_eval_scores = defaultdict(list)

        max_stage = max([k for k in self.optim_config.keys() if isinstance(k, int)]) if self.optim_config.get("max_stage", -1) == -1 else self.optim_config["max_stage"]
        for stage in range(max_stage + 1):
            optim_config = self.optim_config[stage]

            # if "use_best_state" in optim_config and optim_config["use_best_state"]:
            #     self.model.load_best_state()

            # create sampler for training
            if (
                "sampler" not in optim_config
                or optim_config["sampler"].lower() == "random"
            ):
                train_sampler = RandomSampler(self.train_dataset)
            elif optim_config["sampler"].lower() == "ros":
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
            elif optim_config["sampler"].lower() == "rus":
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

            # freeze backbone parameters
            if "freeze_lm" in optim_config and optim_config["freeze_lm"]:
                self.model.freeze_lm()

            # reset classifier parameters
            if "reset_classifier" in optim_config and optim_config["reset_classifier"]:
                self.model.init_classifier()

            train_dataloader = DataLoader(
                self.train_dataset,
                sampler=train_sampler,
                batch_size=optim_config["batch_size"],
            )

            if optim_config["max_steps"] > 0:
                t_total = optim_config["max_steps"]
                optim_config["num_train_epochs"] = (
                    optim_config["max_steps"]
                    // (
                        len(train_dataloader)
                        // optim_config["gradient_accumulation_steps"]
                    )
                    + 1
                )
            else:
                t_total = (
                    len(train_dataloader)
                    // optim_config["gradient_accumulation_steps"]
                    * optim_config["num_train_epochs"]
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
                    "weight_decay": optim_config["weight_decay"],
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
            if optim_config["optimizer"] == "AdamW":
                optimizer = AdamW(
                    optimizer_grouped_parameters,
                    lr=optim_config["learning_rate"],
                    eps=optim_config["adam_epsilon"],
                )
            elif optim_config["optimizer"] == "RMSprop":
                optimizer = RMSprop(
                    optimizer_grouped_parameters,
                    lr=optim_config["learning_rate"],
                    eps=optim_config["adam_epsilon"],
                )
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=optim_config["warmup_steps"],
                num_training_steps=t_total,
            )

            logger.info(
                "***** Running stage %s *****", f"{stage} / {max_stage}"
            )
            logger.info("  Num examples = %d", len(self.train_dataset))
            logger.info("  Num Epochs = %d", optim_config["num_train_epochs"])
            logger.info("  Sampler = %s", optim_config.get("sampler", ""))
            logger.info("  Batch size = %d", optim_config["batch_size"])
            logger.info(
                "  Gradient Accumulation steps = %d",
                optim_config["gradient_accumulation_steps"],
            )
            logger.info("  Total optimization steps = %d", t_total)
            logger.info("  Logging steps = %d", self.train_config["logging_steps"])

            global_step = 0
            tr_loss = 0.0
            self.model.zero_grad()

            train_iterator = trange(
                int(optim_config["num_train_epochs"]),
                desc=f"Train_{self.train_id}. Epoch",
            )

            epoch_times = []
            for epoch, _ in enumerate(train_iterator):
                
                t0 = time.time()
                epoch_iterator = tqdm(train_dataloader, desc="Iteration")
                last_step = len(epoch_iterator)
                for step, batch in enumerate(epoch_iterator):
                    self.model.train()
                    # if self.model_class
                    input_cols = self.input_col_dict[self.model_class]

                    # batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                    inputs = dict()
                    for col in input_cols:
                        # print(batch[col])
                        inputs[col] = batch[col].to(self.device)

                    outputs = self.model(**inputs)
                    loss = outputs[0]

                    if optim_config["gradient_accumulation_steps"] > 1:
                        loss = loss / optim_config["gradient_accumulation_steps"]

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    tr_loss += loss.item()
                    if (step + 1) % optim_config["gradient_accumulation_steps"] == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            optim_config["max_grad_norm"],
                        )

                        optimizer.step()
                        scheduler.step()  # Update learning rate schedule
                        self.model.zero_grad()
                        global_step += 1

                        if (
                            self.train_config["logging_steps"] < 0
                            and step == (last_step - 1)
                        ) or (
                            self.train_config["logging_steps"] > 0
                            and global_step > 0
                            and global_step % self.train_config["logging_steps"] == 0
                        ):
                            self.model.clear_caches()
                            for dev_name, dev_dataset in self.dev_dataset.items():
                                cache_name = f"caches_dev_{dev_name}_stage_{stage}_epoch{epoch}"
                                scores = self.evaluate(
                                    dataset=dev_dataset, filename=f"predictions_dev_{dev_name}_stage_{stage}_epoch{epoch}", 
                                    cache=cache_name
                                )
                                scores["epoch"] = epoch # extra information to save
                                scores["stage"] = stage # extra information to save
                                all_eval_scores[dev_name].append(scores) # to save at the end

                                if (best_eval_scores[dev_name] is None) or (
                                    best_eval_scores[dev_name]["macro_f1"] < scores["macro_f1"]
                                ):
                                    best_eval_scores[dev_name] = scores
                                    best_eval_epoch[dev_name] = epoch
                                    best_eval_step[dev_name] = step
                                    self.model.mark_as_best(dev_name, cache_name=cache_name)

                    if 0 < optim_config["max_steps"] < global_step:
                        epoch_iterator.close()
                        break

                epoch_times.append(time.time() - t0)

                if 0 < optim_config["max_steps"] < global_step:
                    train_iterator.close()
                    break

            logger.info("  Average time per epoch = %d", np.mean(epoch_times))
            logger.info(
                "  Number of samples per second = %d",
                len(self.train_dataset) / np.mean(epoch_times),
            )

            # End of current stage: save the states, and caches of the best model
            for dev_name in self.dev_dataset.keys():
                if (
                    best_eval_scores[dev_name] is not None
                    and best_eval_epoch[dev_name] is not None
                    and (
                        self.global_best_scores is None # TODO: update grid search to enable multiple dev datasets
                        or bestdev_dataset_eval_scores["macro_f1"]
                        > self.global_best_scores["macro_f1"]
                    )
                ):
                    state_filename = f"id{self.train_id}_stage{stage}_dev_{dev_name}_epoch{best_eval_epoch[dev_name]}_step{best_eval_step[dev_name]}.pt"
                    self.model.save_best_state(
                        name=dev_name,
                        output_dir=self.output_dir,
                        filename=state_filename,
                    )
                    self.model.save_caches(output_dir=self.output_dir)
        if self.save_eval_scores:
            for dev_name in self.dev_dataset.keys():
                if len(all_eval_scores[dev_name]) > 0:
                    pd.DataFrame(data=all_eval_scores[dev_name]).to_csv(
                        self.output_dir / f"dev_{dev_name}_scores.csv"
                    )
        return best_eval_epoch, best_eval_step, best_eval_scores, state_filename

    def eval(self):
        # load states
        for name, test_dataset in self.test_dataset.items():
            eval_scores = self.evaluate(
                dataset=test_dataset, filename=f"predictions_test_{name}.csv"
            )
            pd.DataFrame(data=[eval_scores]).to_csv(
                self.output_dir / f"scores_test_{name}.csv"
            )

    def evaluate(self, dataset, filename="", cache=""):
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(
            dataset,
            sampler=eval_sampler,
            batch_size=self.eval_config["batch_size"],
        )

        # Eval!
        logger.info("***** Running evaluation on %s*****", dataset.name)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.eval_config["batch_size"])
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        labels = None
        keys = None

        self.model.eval()
        t0 = time.time()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):

            with torch.no_grad():

                input_cols = self.input_col_dict[self.model_class]
                inputs = dict()
                for col in input_cols:
                    inputs[col] = batch[col].to(self.device)

                outputs = self.model(cache=cache, **inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                labels = inputs["label"].detach().cpu().numpy()
                keys = batch["key"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                labels = np.append(
                    labels, inputs["label"].detach().cpu().numpy(), axis=0
                )
                keys = np.append(keys, batch["key"].detach().cpu().numpy(), axis=0)

        logger.info(
            "  Number of samples per second during inference = %d",
            len(dataset) / (time.time() - t0),
        )
        eval_details_path = self.output_dir / f"{filename}.csv"
        # logger.info("  Average time per epoch = %d", np.mean(epoch_times))
        eval_loss = eval_loss / nb_eval_steps
        results = {"loss": eval_loss}
        preds = np.argmax(preds, axis=1)
        details = dataset.get_df()
        details = join_eval_details(self.data_config, details, preds, keys)
        result = compute_metrics(
            details["pred"], details["label"], docids=details["docid"]
        )
        results.update(result)
        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  {} = {:.4f}".format(key, results[key]))
        if self.save_eval_details:
            details.to_csv(eval_details_path, index=True, encoding="utf-8-sig")
        return results
