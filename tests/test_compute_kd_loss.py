import unittest
import sys
import torch
from torch.nn.functional import mse_loss, kl_div, log_softmax, softmax

from nlp_pipeline.trainer_kd import KDTrainer
# # passed

def compute_kd_old_loss(student_logits, teacher_logits, kd_config):
        soft_student = log_softmax(student_logits / kd_config["kl_T"], dim=-1)
        soft_teacher = softmax(teacher_logits / kd_config["kl_T"], dim=-1)
        return kd_config["kl_T"] ** 2 * kl_div(
                soft_student, soft_teacher, reduction="batchmean"
        )

class TestComputeKD(unittest.TestCase):
    def test_vanilla_KD_kl(self):
        hard_loss = 0
        student_logits = torch.tensor([
            [-2.07827187,  3.96284986, -1.89321542],
            [-1.89834762,  2.62146664, -0.60416764]
        ])
        teacher_logits = torch.tensor([
            [-1.52182424,  3.94843602, -2.65610909],
            [2.79694033, -0.32314146, -2.20684624] 
        ])
        kd_config = {
            "loss_type": "kl",
            "soft_lambda": 1.0,
            "kl_T": 5
        }
        
        loss_new = KDTrainer.compute_kd_loss(hard_loss, student_logits, teacher_logits, kd_config)
        loss_old = compute_kd_old_loss(student_logits, teacher_logits, kd_config)
        # KL loss is torch tensor format scalar
        self.assertTrue(torch.isclose(loss_new, loss_old))
        
    def test_vanilla_KD_mse(self):
        hard_loss = 0
        student_logits = torch.tensor([
            [-2.07827187,  3.96284986, -1.89321542],
            [-1.89834762,  2.62146664, -0.60416764]
        ])
        teacher_logits = torch.tensor([
            [-1.52182424,  3.94843602, -2.65610909],
            [2.79694033, -0.32314146, -2.20684624] 
        ])
        kd_config = {
            "loss_type": "mse",
            "soft_lambda": 1.0,
            "kl_T": 5
        }
        
        loss_result = KDTrainer.compute_kd_loss(hard_loss, student_logits, teacher_logits, kd_config)
        # use other package to calculate
        
        loss_target = torch.mean(torch.mean((student_logits - teacher_logits)**2, axis=1))
        self.assertTrue(torch.isclose(loss_result, loss_target))
        
    def test_dynamic_temperature(self):
        student_logits = torch.tensor([
            [-2.07827187,  3.96284986, -1.89321542],
            [-1.89834762,  2.62146664, -0.60416764] # swap for extreme different, test temp B >Ａ
        ])
        teacher_logits = torch.tensor([
            [-1.52182424,  3.94843602, -2.65610909],
            [2.79694033, -0.32314146, -2.20684624] 
        ])
        kd_config = {
            "loss_type": "kl",
            "soft_lambda": 1.0,
            "kl_T": 5,
            "dtd_type": "flsw",
            "dtd_bias": 10,
            "dtd_flsw_pow": 1 
        }
        
        temp_result = KDTrainer.get_dynamic_temperature(student_logits, teacher_logits, kd_config)
        confident_temp = temp_result[0].item()
        unconfident_temp = temp_result[1].item()
        self.assertGreaterEqual(confident_temp, unconfident_temp)
    