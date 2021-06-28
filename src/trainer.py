# coding=utf-8
import logging
import numpy as np
import copy
import sklearn
import torch
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    WeightedRandomSampler,
)
from torch.optim import RMSprop
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup
from metric import compute_metrics
from torch.utils.tensorboard import SummaryWriter


logger = logging.getLogger(__name__)


def prediction_step(model, batch, args):
    model.eval()
    results = dict()
    with torch.no_grad():
        inputs = dict()
        for col in batch:
            if torch.is_tensor(batch[col]):
                inputs[col] = batch[col].to(args.device).long()
        x = model(
            **inputs,
        )

    results["prediction"] = []
    for x1 in x[1]:
        if isinstance(x1, list):
            results["prediction"].append(list(map(lambda x: args.label_to_id_inv[x], x1)))
        else:
            results["prediction"].append(args.label_to_id_inv[x1])
    # check

    results["logits"] = x[2].cpu().tolist()
    if x[0] is not None:
        results["loss"] = x[0].cpu().tolist()
    else:
        results["loss"] = None
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

    label_ids = []
    predictions = []
    losses = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        results = prediction_step(model, batch, args=args)
        label_ids.extend(batch["label"].cpu().tolist())
        losses.append(results["loss"])
        predictions.extend(results["prediction"])

    labels = []
    for l1, p1 in zip(label_ids, predictions):
        if isinstance(l1, list):
            labels.append(list(map(lambda x: args.label_to_id_inv[x], l1))[:len(p1)])
        else:
            labels.append(args.label_to_id_inv[l1])

    metrics = compute_metrics(task=args.task, labels=labels, predictions=predictions) 
    metrics['loss'] = np.mean(losses)
    metrics['dataset'] = eval_dataset.dataset
    for m in metrics:
        logger.info("  %s = %s", m, str(metrics[m]))

    eval_dataset.insert_predictions(predictions)
    return metrics


class Trainer:
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
        self.model_dir = args.model_dir

        self.best_score = None
        self.best_model_state = None
        self.non_increase_cnt = 0
        self.early_stop = self.train_config.get('early_stop', None)
        self.final_model = self.train_config.get('final_model', "last")
        self.tensorboard_writer = SummaryWriter(self.args.tensorboard_dir)

    def create_optimizer_and_scheduler(self, n):
        if self.model_config["max_steps"] > 0:
            t_total = self.model_config["max_steps"]
            self.model_config["num_train_epochs"] = (
                self.model_config["max_steps"]
                // (n // self.model_config["gradient_accumulation_steps"])
                + 1
            )
        else:
            t_total = (
                n
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

        scheduler = self.model_config.get('scheduler', None)
        if scheduler=='linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(self.model_config["warmup_steps"]),
                num_training_steps=t_total,
            )

        return optimizer, scheduler

    def train(self):
        dataloader = DataLoader(
            self.train_dataset,
            sampler=RandomSampler(self.train_dataset), 
            batch_size=self.train_config["batch_size"],
            # collate_fn=self.train_dataset.pad_collate,
        )
        optimizer, scheduler = self.create_optimizer_and_scheduler(n=len(dataloader))
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.model_config["num_train_epochs"])
        logger.info("  Sampler = %s", self.model_config.get("sampler", ""))
        logger.info("  Batch size = %d", self.train_config["batch_size"])
        logger.info(
            "  Gradient Accumulation steps = %d",
            self.model_config["gradient_accumulation_steps"],
        )
        
        train_iterator = trange(
            int(self.model_config["num_train_epochs"]),
            desc=f"Epoch",
        )
        global_step = 0
        for epoch, _ in enumerate(train_iterator):
            self.model.zero_grad()
            self.model.train()
            epoch_iterator = tqdm(dataloader, desc="Iteration")
            last_step = len(epoch_iterator)
            for step, batch in enumerate(epoch_iterator):
                inputs = dict()
                for col in batch:
                    inputs[col] = batch[col].to(self.device).long()
                outputs = self.model(**inputs)
                loss = outputs[0]
                logits = outputs[1]
                loss.backward()
                loss = loss.tolist()
                self.tensorboard_writer.add_scalar('Loss/train', loss, global_step)
                if (step + 1) % self.model_config["gradient_accumulation_steps"] == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.model_config["max_grad_norm"],
                    )
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step() 
                    self.model.zero_grad()
                epoch_iterator.set_postfix({"tr_loss": np.mean(loss)})
                global_step += 1

            self.on_epoch_end(epoch)
            if self.early_stop is not None and self.non_increase_cnt >= self.early_stop:
                break
        self.on_training_end()

    def on_epoch_end(self, epoch):
        logger.info(f"***** Epoch end: {epoch} *****")
        metrics = evaluate(
            model=self.model,
            eval_dataset=self.dev_dataset,
            args=self.args,
        )
        # write train loss & dev metrics on tensorboard
        if self.final_model=="best":
            opt_metric = self.train_config.get('optimization_metric', "macro_f1")
            if self.best_score is None or self.best_score < metrics[opt_metric]:
                self.best_score = metrics[opt_metric]
                self.best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                self.non_increase_cnt += 1
        self.tensorboard_writer.add_scalar(f'{opt_metric}/dev', metrics[opt_metric], epoch)

    def on_training_end(self):
        logger.info("***** Training end *****")
        out_path = (
            self.model_dir / f"model.pt"
        )
        if self.final_model=="best" and self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        logger.info("  Model path = %s", str(out_path))
        torch.save(self.model, out_path)
        self.tensorboard_writer.close()

