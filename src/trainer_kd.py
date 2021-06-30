import logging
import os 
import random
import torch
import pickle
from tqdm import tqdm

from torch.nn.functional import mse_loss, kl_div, log_softmax, softmax
from torch.utils.data import DataLoader, RandomSampler

from trainer import Trainer, prediction_step
from utils import make_batches


logger = logging.getLogger(__name__)


def get_logits(model, dataset, teacher_args, student_args):
    # load
    # chang to student model dir
    logits_path = student_args.model_dir / f"logits_{dataset.dataset}.pkl"
    if os.path.isfile(logits_path):
        logger.info("***** Loading logits *****")
        logger.info("  Logits path = %s", str(logits_path))
        logits = pickle.load(open(logits_path, 'rb'))
    else:
        logger.info("***** Generating logits *****")
        dataloader = DataLoader(
            dataset,
            shuffle=False, 
            batch_size=teacher_args.eval_config["batch_size"],
        ) 
        logits = []
        for batch in tqdm(dataloader, desc="Getting logits"):
            results = prediction_step(model, batch, args=teacher_args)
            if len(logits)==0:
                logits = results["logits"]
            else:
                logits.extend(results["logits"])
        dataset.insert_skipped_rows(logits)
        # save
        pickle.dump(logits, open(logits_path, 'wb'))
    return logits


class KDTrainer(Trainer):
    def __init__(
        self,
        model, 
        train_dataset, 
        dev_dataset, 
        unlabeled_dataset, 
        args
    ):
        super(KDTrainer, self).__init__(model=model, train_dataset=train_dataset, dev_dataset=dev_dataset, args=args)
        self.unlabeled_dataset = unlabeled_dataset
        self.kd_config = args.kd_config
        

    @staticmethod
    def compute_kd_loss(hard_loss, student_logits, teacher_logits, kd_config):
        if hard_loss is None:
            hard_loss = 0
        loss_type = kd_config['loss_type'] 
        soft_lambda = kd_config['soft_lambda'] 
        kl_T = kd_config['kl_T'] 

        if loss_type=="mse":
            # https://pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html#torch.nn.functional.mse_loss
            # print(student_logits.size())
            # print(teacher_logits.size())
            soft_loss = mse_loss(
                input=student_logits, 
                target=teacher_logits, 
                reduction="mean"
            )
        elif loss_type=="kl":
            soft_student = log_softmax(student_logits / kl_T, dim=-1)
            soft_teacher = softmax(teacher_logits / kl_T, dim=-1)
            soft_loss = kl_T**2 * kl_div(soft_student, soft_teacher, reduction='batchmean')
        else:
            raise ValueError(f"Expected knowledge distillation loss type 'mse' or 'kl', but got {kd_type}")
        loss = (1 - soft_lambda) * hard_loss + soft_lambda * soft_loss
        return loss

    def train(self):
        logger.info("***** Running KD training *****")
        logger.info("  Num examples (train) = %d", len(self.train_dataset))
        logger.info("  Num examples (unlabeled)= %d", len(self.unlabeled_dataset))
        logger.info("  Num Epochs = %d", self.model_config["num_train_epochs"])
        logger.info("  Batch size = %d", self.train_config["batch_size"])

        batch_size = self.train_config['batch_size']
        total_epochs = self.model_config["num_train_epochs"]
        optimizer, scheduler = self.create_optimizer_and_scheduler(n=len(self.unlabeled_dataset))
        
        n_step_tr = 0
        n_step_ul = 0
        for epoch in range(total_epochs):
            self.model.zero_grad()
            self.model.train()

            dataloader_ul = DataLoader(
                self.unlabeled_dataset,
                batch_size=batch_size,
                shuffle=True
            )

            dataloader_tr = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=True
            )

            log_steps = self.train_config.get('log_steps', 1)

            for batch in tqdm(dataloader_tr):

                inputs = dict()
                for col in batch:
                    if torch.is_tensor(batch[col]):
                        inputs[col] = batch[col].to(self.device).long()
                outputs = self.model(**inputs)
                
                hard_loss = outputs[0]
                student_logits = outputs[2]
                teacher_logits = batch['teacher_logit'].to(self.device)

                loss = self.compute_kd_loss(
                    hard_loss=hard_loss,
                    student_logits=student_logits, 
                    teacher_logits=teacher_logits,
                    kd_config=self.kd_config 
                )

                loss.backward()
                optimizer.step()
                self.model.zero_grad()
                if n_step_tr % log_steps==0:
                    self.tensorboard_writer.add_scalar('Loss/train', loss.tolist(), n_step_tr)
                n_step_tr += 1

            for batch in tqdm(dataloader_ul):

                inputs = dict()
                for col in batch:
                    if torch.is_tensor(batch[col]):
                        inputs[col] = batch[col].to(self.device).long()

                outputs = self.model(**inputs)
                hard_loss = outputs[0]
                student_logits = outputs[2]
                teacher_logits = batch['teacher_logit'].to(self.device)

                loss = self.compute_kd_loss(
                    hard_loss=hard_loss,
                    student_logits=student_logits, 
                    teacher_logits=teacher_logits,
                    kd_config=self.kd_config 
                )
                
                loss.backward()
                optimizer.step()
                self.model.zero_grad()
                if n_step_ul % log_steps==0:
                    self.tensorboard_writer.add_scalar('Loss/unlabeled', loss.tolist(), n_step_ul)
                n_step_ul += 1
            self.on_epoch_end(epoch)
        self.on_training_end()
        




