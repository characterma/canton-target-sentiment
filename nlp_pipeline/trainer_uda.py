import logging
from tqdm import tqdm
from tqdm import trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from nlp_pipeline.trainer import Trainer


logger = logging.getLogger(__name__)


class TrainerUDA(Trainer):
    def __init__(self, model, labeled_dataset, unlabeled_dataset, dev_dataset, args):
        super().__init__(model, labeled_dataset, dev_dataset, args)
        self.unlabeled_dataset = unlabeled_dataset
        self.labeled_dataset = labeled_dataset
        self.kl_loss = nn.KLDivLoss(reduction='none')
        self.sup_criterion = nn.CrossEntropyLoss(reduction='none')

        self.tsa = self.args.uda_config.get("tsa", 'linear_schedule')
        self.total_steps = self.args.uda_config.get("total_steps", 10000)
        self.eval_steps = self.args.uda_config.get("eval_steps", 1000)
        self.uda_coeff = self.args.uda_config.get("uda_coeff", 1)
        self.uda_confidence_thresh = self.args.uda_config.get("uda_confidence_thresh", 0.45)
        self.uda_softmax_temp = self.args.uda_config.get("uda_softmax_temp", 0.85)

    def train(self):
        """
            ref:
                https://github.com/SanghunYun/UDA_pytorch/blob/master/train.py
                https://github.com/SanghunYun/UDA_pytorch/blob/master/main.py
        """
        labeled_dataloader = iter(DataLoader(
            self.labeled_dataset,
            sampler=RandomSampler(self.labeled_dataset),
            batch_size=self.train_config["batch_size"],
            collate_fn=self.labeled_dataset.collate_fn,
        ))

        unlabeled_dataloader = iter(DataLoader(
            self.unlabeled_dataset,
            sampler=RandomSampler(self.unlabeled_dataset),
            batch_size=self.train_config["batch_size"],
            collate_fn=self.unlabeled_dataset.collate_fn,
        ))

        optimizer, scheduler = self.create_optimizer_and_scheduler(n=len(unlabeled_dataloader))
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.model_config["num_train_epochs"])
        logger.info("  Sampler = %s", self.model_config.get("sampler", ""))
        logger.info("  Batch size = %d", self.train_config["batch_size"])

        if self.total_steps is None:
            self.total_steps = len(train_iterator) * len(unlabeled_dataloader)

        global_step = 0
        self.model.zero_grad()
        self.model.train()
        # epoch_iterator = tqdm(unlabeled_dataloader, desc="Iteration")
        for step in tqdm(range(self.total_steps), desc="Steps"):
            unlabeled_batch = next(unlabeled_dataloader, None)
            if unlabeled_batch is None:
                unlabeled_dataloader = iter(DataLoader(
                    self.unlabeled_dataset,
                    sampler=RandomSampler(self.unlabeled_dataset),
                    batch_size=self.train_config["batch_size"],
                    collate_fn=self.unlabeled_dataset.collate_fn,
                ))
                unlabeled_batch = next(unlabeled_dataloader, None)

            labeled_batch = next(labeled_dataloader, None)
            if labeled_batch is None:
                labeled_dataloader = iter(DataLoader(
                    self.labeled_dataset,
                    sampler=RandomSampler(self.labeled_dataset),
                    batch_size=self.train_config["batch_size"],
                    collate_fn=self.labeled_dataset.collate_fn,
                ))
                labeled_batch = next(labeled_dataloader, None)

            supervised_loss = self.get_supervised_loss(labeled_batch, global_step)
            consistency_loss = self.get_consistency_loss(unlabeled_batch)
            final_loss = supervised_loss + self.uda_coeff * consistency_loss
            final_loss.backward()
            optimizer.step()
            self.model.zero_grad()
            global_step += 1

            if global_step > 0 and global_step % self.eval_steps == 0:
                self.on_epoch_end(global_step)

        self.on_training_end()

    def get_tsa_threshold(self, global_step, start, end):
        training_progress = torch.tensor(float(global_step) / float(self.total_steps))
        if self.tsa == 'linear_schedule':
            threshold = training_progress
        elif self.tsa == 'exp_schedule':
            scale = 5
            threshold = torch.exp((training_progress - 1) * scale)
        elif self.tsa == 'log_schedule':
            scale = 5
            threshold = 1 - torch.exp((-training_progress) * scale)
        output = threshold * (end - start) + start
        return output.to(self.device)

    def get_supervised_loss(self, batch, global_step):
        inputs = dict()
        for k, v in batch.items():
            inputs[k] = v.to(self.device).long()
            sup_size = v.shape[0]     

        outputs = self.model(**inputs)

        labels = inputs['label']
        logits = outputs['logits']

        supervised_loss = self.sup_criterion(logits, labels)  
        if self.tsa:
            tsa_thresh = self.get_tsa_threshold(global_step, start= 1. / logits.shape[-1], end=1)
            larger_than_threshold = torch.exp(-supervised_loss) > tsa_thresh
            loss_mask = torch.ones_like(labels, dtype=torch.float32) * (1 - larger_than_threshold.type(torch.float32))
            supervised_loss = torch.sum(supervised_loss * loss_mask, dim=-1) / torch.max(torch.sum(loss_mask, dim=-1), torch.tensor(1.).to(self.device))
        else:
            supervised_loss = torch.mean(supervised_loss)

        return supervised_loss

    def get_consistency_loss(self, batch):
        batch_non_da = dict()
        batch_da = dict()
        uda_softmax_temp = self.uda_softmax_temp if self.uda_softmax_temp > 0 else 1.

        for k, v in batch.items():
            if k.endswith('_da'):
                batch_da[k[:-3]] = v.to(self.device).long()
            else:
                batch_non_da[k] = v.to(self.device).long()

        with torch.no_grad():
            outputs_non_da = self.model(**batch_non_da)
            prob_non_da = F.softmax(outputs_non_da['logits'], dim=-1)

            # confidence-based masking
            if self.uda_confidence_thresh != -1:
                consistency_loss_mask = torch.max(prob_non_da, dim=-1)[0] > self.uda_confidence_thresh
                consistency_loss_mask = consistency_loss_mask.type(torch.float32)
            else:
                consistency_loss_mask = torch.ones(len(logits) - sup_size, dtype=torch.float32)
            consistency_loss_mask = consistency_loss_mask.to(self.args.device)

        outputs_da = self.model(**batch_da)
        prob_da = F.softmax(outputs_da['logits'] / uda_softmax_temp, dim=-1)

        consistency_loss = torch.sum(
            self.kl_loss(prob_da, prob_non_da), 
            dim=-1
        )
        consistency_loss = torch.sum(consistency_loss * consistency_loss_mask, dim=-1) / torch.max(torch.sum(consistency_loss_mask, dim=-1), torch.tensor(1.).to(self.args.device))
        return consistency_loss
