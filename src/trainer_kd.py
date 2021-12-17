import logging
import os
import torch
import pickle
from tqdm import tqdm

from torch.nn.functional import mse_loss, kl_div, log_softmax, softmax
from torch.utils.data import DataLoader

from trainer import Trainer, prediction_step


logger = logging.getLogger(__name__)


def get_logits(model, dataset, teacher_args, student_args):
    logits_path = student_args.model_dir / f"logits_{dataset.dataset}.pkl"
    if os.path.isfile(logits_path):
        logger.info("***** Loading logits *****")
        logger.info("  Logits path = %s", str(logits_path))
        logits = pickle.load(open(logits_path, "rb"))
    else:
        logger.info("***** Generating logits *****")
        dataloader = DataLoader(
            dataset, shuffle=False, batch_size=teacher_args.eval_config["batch_size"]
        )
        logits = []
        for batch in tqdm(dataloader, desc="Getting logits"):
            results = prediction_step(model, batch, args=teacher_args)
            if len(logits) == 0:
                logits = results["probabilities"]
            else:
                logits.extend(results["probabilities"])
        dataset.insert_skipped_samples(logits)
        # save
        pickle.dump(logits, open(logits_path, "wb"))
    return logits


class KDTrainer(Trainer):
    def __init__(self, model, train_dataset, dev_dataset, unlabeled_dataset, args):
        super(KDTrainer, self).__init__(
            model=model, train_dataset=train_dataset, dev_dataset=dev_dataset, args=args
        )
        self.unlabeled_dataset = unlabeled_dataset
        self.kd_config = args.kd_config

    @staticmethod
    def get_dynamic_temperature(student_logits, teacher_logits, kd_config):
        '''
        input:
        - soft_student_logits: torch.tensor
        - soft_teacher_logits: torch.tensor
        - kd_config: dict

        output:
        - torch.tensor (1 dimension)
        '''
        types = kd_config['dtd_type']
        dtd_bias = kd_config['dtd_bias']
        dtd_flsw_pow = kd_config['dtd_flsw_pow']
        initial_temp = kd_config['kl_T']

        if types == 'NA':
            return torch.tensor(
                [[initial_temp for i in range(teacher_logits.shape[0])]]
            ).T.to(student_logits.device)

        wx = []
        soft_student_logits = softmax(student_logits, dim=-1)
        soft_teacher_logits = softmax(teacher_logits, dim=-1)
        for i in range(soft_teacher_logits.shape[0]):
            if types == 'flsw':
                # cos = nn.CosineSimilarity(dim = 0, eps=1e-6)
                wx.append((1- torch.dot(soft_student_logits[i], soft_teacher_logits[i]))**dtd_flsw_pow)
            elif types == 'cwsm':
                wx.append(1/torch.max(soft_student_logits[i]))
            else:
                raise ValueError(f"Expected knowledge distillation loss type 'NA' or 'flsw' or 'cwsm'")

        kl_temp = []
        for i in range(soft_teacher_logits.shape[0]):
            dy_temp = initial_temp + (sum(wx)/soft_teacher_logits.shape[0] - wx[i]) * dtd_bias
            dy_temp = dy_temp if dy_temp >= 1 else 1
            kl_temp.append(dy_temp)
        return torch.tensor([kl_temp]).T.to(student_logits.device)


    @staticmethod
    def compute_kd_loss(hard_loss, student_logits, teacher_logits, kd_config):
        if hard_loss is None:
            hard_loss = 0
        loss_type = kd_config["loss_type"]
        soft_lambda = kd_config["soft_lambda"]
        kl_T = kd_config["kl_T"]

        if loss_type == "mse":
            # https://pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html#torch.nn.functional.mse_loss
            # print(student_logits.size())
            # print(teacher_logits.size())
            soft_loss = mse_loss(
                input=student_logits, target=teacher_logits, reduction="mean"
            )
        elif loss_type == "kl":
            kl_T = KDTrainer.get_dynamic_temperature(student_logits, teacher_logits, kd_config)
            soft_student = log_softmax(student_logits / kl_T, dim=-1)
            soft_teacher = softmax(teacher_logits / kl_T, dim=-1)
            # sum of classes KL -> soften logit -> batchmean
            soft_loss = (kl_T ** 2 * kl_div(
                soft_student, soft_teacher, reduction="none"
            ).sum(axis=1)).mean()
            # old method
            # soft_loss = kl_T ** 2 * kl_div(
            #     soft_student, soft_teacher, reduction="batchmean"
            # )
        else:
            raise ValueError(f"Expected knowledge distillation loss type 'mse' or 'kl'")
        loss = (1 - soft_lambda) * hard_loss + soft_lambda * soft_loss
        return loss

    def train(self):
        logger.info("***** Running KD training *****")
        logger.info("  Num examples (train) = %d", len(self.train_dataset))
        logger.info("  Num examples (unlabeled)= %d", len(self.unlabeled_dataset) if self.unlabeled_dataset is not None else 0)
        logger.info("  Num Epochs = %d", self.model_config["num_train_epochs"])
        logger.info("  Batch size = %d", self.train_config["batch_size"])

        batch_size = self.train_config["batch_size"]
        total_epochs = self.model_config["num_train_epochs"]
        optimizer, scheduler = self.create_optimizer_and_scheduler(
            n=len(self.train_dataset) + len(self.unlabeled_dataset) if self.unlabeled_dataset is not None else len(self.train_dataset)
        )

        n_step_tr = 0
        n_step_ul = 0
        n_step_tr_log = 0
        n_step_ul_log = 0

        for epoch in range(total_epochs):
            self.model.zero_grad()
            self.model.train()

            dataloader_tr = DataLoader(
                self.train_dataset, batch_size=batch_size, shuffle=True
            )

            for batch in tqdm(dataloader_tr):

                inputs = dict()
                for col in batch:
                    if torch.is_tensor(batch[col]):
                        inputs[col] = batch[col].to(self.device).long()
                outputs = self.model(**inputs)

                hard_loss = outputs.loss
                student_logits = outputs['logits']
                teacher_logits = batch["teacher_logit"].to(self.device)

                loss = self.compute_kd_loss(
                    hard_loss=hard_loss,
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    kd_config=self.kd_config,
                )

                loss.backward()
                optimizer.step()
                self.model.zero_grad()
                # if n_step_tr % log_steps==0:
                self.tensorboard_writer.add_scalar(
                    "Loss/train", loss.tolist(), n_step_tr_log
                )
                # n_step_tr_log += 1
                n_step_tr += 1
                
            if self.unlabeled_dataset is not None:
                dataloader_ul = DataLoader(
                    self.unlabeled_dataset, batch_size=batch_size, shuffle=True
                )

                for batch in tqdm(dataloader_ul):

                    inputs = dict()
                    for col in batch:
                        if torch.is_tensor(batch[col]):
                            inputs[col] = batch[col].to(self.device).long()

                    outputs = self.model(**inputs)
                    hard_loss = outputs.loss
                    student_logits = outputs['logits']
                    teacher_logits = batch["teacher_logit"].to(self.device)

                    loss = self.compute_kd_loss(
                        hard_loss=hard_loss,
                        student_logits=student_logits,
                        teacher_logits=teacher_logits,
                        kd_config=self.kd_config,
                    )

                    loss.backward()
                    optimizer.step()
                    self.model.zero_grad()
                    # if n_step_ul % log_steps==0:
                    self.tensorboard_writer.add_scalar(
                        "Loss/unlabeled", loss.tolist(), n_step_ul_log
                    )
                    # n_step_ul_log += 1
                    n_step_ul += 1
            self.on_epoch_end(epoch)
        self.on_training_end()
