# coding=utf-8
import logging
import numpy as np
import copy
import sklearn
import torch
import time
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
from torch.optim import RMSprop
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from nlp_pipeline.metric import compute_metrics
from nlp_pipeline.adversarial import get_adversarial_class
from nlp_pipeline.loss import FocalLoss


logger = logging.getLogger(__name__)


def prediction_step(model, batch, args):
    model.eval()
    results = dict()
    with torch.no_grad():
        inputs = dict()
        for col in batch:
            if torch.is_tensor(batch[col]):
                inputs[col] = batch[col].to(args.device).long()
        x = model(**inputs)

    results["prediction_id"] = []
    results["prediction"] = []

    if isinstance(x['prediction'], torch.Tensor):
        predictions = x['prediction'].tolist()
    else:
        predictions = x['prediction'] 

    for p in predictions:
        if isinstance(p, list):
            results["prediction_id"].append(p)
            results["prediction"].append(
                list(map(lambda y: args.label_to_id_inv[y], p))
            )
        else:
            results["prediction_id"].append(p)
            results["prediction"].append(args.label_to_id_inv[p])

    results["probabilities"] = F.softmax(x["logits"], dim=-1).cpu().tolist()
    if x.loss is not None:
        results["loss"] = x.loss.cpu().tolist()
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
        collate_fn=eval_dataset.collate_fn,
    )

    label_ids = []
    predictions = []
    prediction_ids = []
    probabilities = []
    losses = []

    has_label = False
    n_samples = 0
    total_time = 0
    for batch in tqdm(dataloader, desc="Evaluating"):
        t0 = time.time()
        results = prediction_step(model, batch, args=args)
        n_samples += len(batch)
        total_time += (time.time() - t0)
        losses.append(results["loss"])
        predictions.extend(results["prediction"])
        prediction_ids.extend(results["prediction_id"])
        probabilities.extend(results["probabilities"])

        if "label" in batch:
            has_label = True
            label_ids.extend(batch["label"].cpu().tolist())

    if has_label:
        metrics = compute_metrics(args=args, label_ids=label_ids, predictions=predictions)
        metrics["loss"] = np.mean(losses)
        metrics["dataset"] = eval_dataset.dataset
        metrics["samples_per_second"] = n_samples / total_time
        for m in metrics:
            logger.info("  %s = %s", m, str(metrics[m]))
    else:
        metrics = {}

    eval_dataset.insert_diagnosis_column(predictions, "prediction")
    eval_dataset.insert_diagnosis_column(prediction_ids, "prediction_id")
    eval_dataset.insert_diagnosis_column(probabilities, "probabilities")
    return metrics


class Trainer:
    def __init__(self, model, train_dataset, dev_dataset, args):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.model = model
        self.device = args.device if torch.cuda.is_available() else "cpu"

        self.model_config = args.model_config
        self.train_config = args.train_config
        self.mlops_config = args.mlops_config
        self.model_dir = args.model_dir

        self.best_score = None
        self.best_model_state = None
        self.non_increase_cnt = 0

        self.warmup_steps = self.model_config.get("warmup_steps", 0)

        self.early_stop = self.train_config.get("early_stop", None)

        self.enable_focal_loss = self.train_config.get("enable_focal_loss", False)
        self.focal_loss_gamma = self.train_config.get("focal_loss_gamma", 2)
        self.focal_loss_reduction = self.train_config.get("focal_loss_reduction", "mean")

        self.enable_model_ema = self.train_config.get("enable_model_ema", False)
        self.model_ema_alpha = self.train_config.get("model_ema_alpha", 0.5)
        self.model_ema_steps = self.train_config.get("model_ema_steps", 100)

        self.enable_adversarial = self.train_config.get("enable_adversarial", False)
        self.adversarial_class = self.train_config.get("adversarial_class", 'PGD')
        self.adversarial_k = self.train_config.get("adversarial_k", 3)
        self.adversarial_param_names = self.train_config.get("adversarial_param_names", ['emb.'])
        self.adversarial_alpha = self.train_config.get("adversarial_alpha", 1)
        self.adversarial_epsilon = self.train_config.get("adversarial_epsilon", 0.3)

        self.initialize_focal_loss()
        self.initialize_model_ema()
        self.initialize_adversarial()

        self.final_model = self.train_config.get("final_model", "last")
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

        scheduler = self.model_config.get("scheduler", None)
        if scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(self.warmup_steps),
                num_training_steps=t_total,
            )

        return optimizer, scheduler


    def compute_kl_loss(self, p, q, pad_mask=None):
        
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
        
        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.mean()
        q_loss = q_loss.mean()

        loss = (p_loss + q_loss) / 2
        return loss

    def initialize_focal_loss(self):
        if self.enable_focal_loss:
            self.focal_loss = FocalLoss(
                gamma=self.focal_loss_gamma, 
                reduction=self.focal_loss_reduction
            )
        else:
            self.focal_loss = None

    def initialize_model_ema(self):
        if self.enable_model_ema:
            from nlp_pipeline.ema import ExponentialMovingAverage
            self.model_ema = ExponentialMovingAverage(
                self.model, 
                device=self.device, 
                decay=1.0 - self.model_ema_alpha
            )
        else:
            self.model_ema = None

    def initialize_adversarial(self):
        if self.enable_adversarial:
            self.adversarial = get_adversarial_class(self.adversarial_class)(
                self.model, 
                self.adversarial_param_names
            )
        else:
            self.adversarial = None

    def train(self):
        dataloader = DataLoader(
            self.train_dataset,
            sampler=RandomSampler(self.train_dataset),
            batch_size=self.train_config["batch_size"],
            collate_fn=self.train_dataset.collate_fn,
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
            int(self.model_config["num_train_epochs"]), desc=f"Epoch"
        )
        log_steps = self.train_config.get("log_steps", 1)
        r_drop_factor = self.train_config.get("r_drop_factor", 0)
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

                if self.focal_loss is not None:
                    loss = self.focal_loss(
                        input=outputs['logits'], 
                        target=inputs['label']
                    )
                else:
                    loss = outputs['loss']

                if r_drop_factor > 0:
                    outputs2 = self.model(**inputs)
                    logits1 = outputs['logits']
                    logits2, loss2 = outputs2['logits'], outputs2['loss']
                    kl_loss = self.compute_kl_loss(logits1, logits2)
                    loss = 0.5 * (loss + loss2) + r_drop_factor * kl_loss

                loss.backward()
                self.run_adversarial(inputs)

                loss = loss.tolist()
                if global_step % log_steps == 0:
                    self.tensorboard_writer.add_scalar("Loss/train", loss, global_step)
                    if self.mlops_config.get("neptune") and self.mlops_config["neptune"]["log"]:
                         self.mlops_config["neptune"]["run"]['Loss/train'].log(loss)
                if (step + 1) % self.model_config["gradient_accumulation_steps"] == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.model_config["max_grad_norm"]
                    )
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    self.model.zero_grad()

                self.update_model_ema(global_step)
                epoch_iterator.set_postfix({"tr_loss": np.mean(loss)})
                global_step += 1

            self.on_epoch_end(epoch)
            if self.early_stop is not None and self.non_increase_cnt >= self.early_stop:
                break
        self.on_training_end()

    def update_model_ema(self, step):
        if self.model_ema and step % self.model_ema_steps == 0:
            self.model_ema.update_parameters(self.model)
            if step < self.warmup_steps:
                # Reset ema buffer to keep copying weights during warmup period
                self.model_ema.n_averaged.fill_(0)

    def run_adversarial(self, batch):
        if self.adversarial:
            if self.adversarial_class=='PGD':
                self.adversarial.backup_grad()
                for t in range(self.adversarial_k):
                    self.adversarial.attack(
                        is_first_attack=(t==0), 
                        epsilon=self.adversarial_epsilon, 
                        alpha=self.adversarial_alpha,
                    ) 
                    if t != self.adversarial_k - 1:
                        self.model.zero_grad()
                    else:
                        self.adversarial.restore_grad()
                    
                    outputs = self.model(**batch)
                    loss = outputs['loss']
                    loss.backward() 
                self.adversarial.restore() 
            else:
                self.adversarial.attack(epsilon=self.adversarial_epsilon)
                outputs = self.model(**batch)
                loss = outputs['loss']
                loss.backward() 
                self.adversarial.restore() 

    def update_best_model(self, model, metrics):
        if self.final_model == "best":
            opt_metric = self.train_config.get("optimization_metric", "macro_f1")
            if self.best_score is None or self.best_score < metrics[opt_metric]:
                self.best_score = metrics[opt_metric]
                self.best_model_state = copy.deepcopy(model.state_dict())
            else:
                self.non_increase_cnt += 1
                
    def on_epoch_end(self, epoch):
        logger.info(f"***** Epoch end: {epoch} *****")
        metrics = evaluate(
            model=self.model, eval_dataset=self.dev_dataset, args=self.args
        )
        self.update_best_model(self.model, metrics)
        if self.model_ema:
            metrics_ema = evaluate(
                model=self.model_ema, eval_dataset=self.dev_dataset, args=self.args
            )
            self.update_best_model(self.model_ema, metrics_ema)

        for metric, value in metrics.items():
            if type(value) in [int, float, str]:
                try:
                    value = float(value)
                    self.tensorboard_writer.add_scalar(
                        f"dev/{metric}", float(value), epoch
                    )
                    if self.mlops_config.get("neptune") and self.mlops_config["neptune"]["log"]:
                         self.mlops_config["neptune"]["run"][f"dev/{metric}"].log(float(value))
                except Exception as e:
                    continue

    def on_training_end(self):
        logger.info("***** Training end *****")
        out_path = self.model_dir / f"model.pt"
        if self.final_model == "best" and self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        logger.info("  Model path = %s", str(out_path))
        torch.save(self.model.state_dict(), out_path)
        self.tensorboard_writer.close()
