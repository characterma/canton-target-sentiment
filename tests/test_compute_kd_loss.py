import unittest
import sys
import torch

sys.path.append("../src/")
from torch.nn.functional import mse_loss, kl_div, log_softmax, softmax
from trainer_kd import KDTrainer

class TestComputeKD(unittest.TestCase):
    def test_vanilla_KD_kl(self):
        hard_loss = 0
        student_logits = torch.tensor([
            [-2.07827187,  3.96284986, -1.89321542],
            [-1.89834762,  2.62146664, -0.60416764]
        ])
        teacher_logits = torch.tensor([
            [-1.52182424,  3.94843602, -2.65610909],
            [-2.20684624,  2.79694033, -0.32314146]
        ])
        kd_config = {
            "loss_type": "kl",
            "soft_lambda": 1.0,
            "kl_T": 5
        }
        
        loss_new = KDTrainer.compute_kd_loss(hard_loss, student_logits, teacher_logits, kd_config)
        soft_student = log_softmax(student_logits / kd_config["kl_T"], dim=-1)
        soft_teacher = softmax(teacher_logits / kd_config["kl_T"], dim=-1)
        loss_old = kd_config["kl_T"] ** 2 * kl_div(
            soft_student, soft_teacher, reduction="batchmean"
        )
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
            [-2.20684624,  2.79694033, -0.32314146]
        ])
        kd_config = {
            "loss_type": "mse",
            "soft_lambda": 1.0,
            "kl_T": 5
        }
        
        loss_result = KDTrainer.compute_kd_loss(hard_loss, student_logits, teacher_logits, kd_config)
        loss_target = mse_loss(
                input=student_logits, target=teacher_logits, reduction="mean"
        )
        self.assertEqual(loss_result, loss_target)
        
    def test_dynamic_temperature(self):
        hard_loss = 0
        student_logits = torch.tensor([
            [-2.07827187,  3.96284986, -1.89321542],
            [-1.89834762,  2.62146664, -0.60416764]
        ])
        teacher_logits = torch.tensor([
            [-1.52182424,  3.94843602, -2.65610909],
            [-2.20684624,  2.79694033, -0.32314146]
        ])
        kd_config = {
            "loss_type": "kl",
            "soft_lambda": 1.0,
            "kl_T": 5,
            "dtd_type": "flsw",
            "dtd_bias": 10,
            "dtd_flsw_pow": 2 
        }
        
        temp_result = KDTrainer.get_dynamic_temperature(student_logits, teacher_logits, kd_config)
        
        soft_student_logits = softmax(student_logits, dim=-1)
        soft_teacher_logits = softmax(teacher_logits, dim=-1)
        wx = []
        for i in range(soft_teacher_logits.shape[0]):
            uncertainty_score = 1 - torch.dot(soft_student_logits[i], soft_teacher_logits[i])
            self.assertLessEqual(uncertainty_score.item(), 2)
            self.assertGreaterEqual(uncertainty_score.item(), 0)
            wx.append(uncertainty_score**kd_config['dtd_flsw_pow'])
        kl_temp = []
        for i in range(soft_teacher_logits.shape[0]):
            dy_temp = kd_config['kl_T'] + (sum(wx)/soft_teacher_logits.shape[0] - wx[i]) * kd_config['dtd_bias']
            dy_temp = dy_temp if dy_temp >= 1 else 1
            kl_temp.append(dy_temp)
        temp_target = torch.tensor([kl_temp]).T.to(student_logits.device)
            
        self.assertTrue(torch.allclose(temp_result, temp_target))
        
    def test_dynamic_temperature_distillation(self):
        hard_loss = 0
        student_logits = torch.tensor([
            [-2.07827187,  3.96284986, -1.89321542],
            [-1.89834762,  2.62146664, -0.60416764]
        ])
        teacher_logits = torch.tensor([
            [-1.52182424,  3.94843602, -2.65610909],
            [-2.20684624,  2.79694033, -0.32314146]
        ])
        kd_config = {
            "loss_type": "kl",
            "soft_lambda": 1.0,
            "kl_T": 5,
            "dtd_type": "flsw",
            "dtd_bias": 10,
            "dtd_flsw_pow": 2 
        }
        
        loss_result = KDTrainer.compute_kd_loss(hard_loss, student_logits, teacher_logits, kd_config)   
        
        kl_T = KDTrainer.get_dynamic_temperature(student_logits, teacher_logits, kd_config)
        soft_student = log_softmax(student_logits / kl_T, dim=-1)
        soft_teacher = softmax(teacher_logits / kl_T, dim=-1)
        # sum of classes KL -> soften logit -> batchmean
        loss_target = (kl_T ** 2 * kl_div(
            soft_student, soft_teacher, reduction="none"
        ).sum(axis=1)).mean()
        
        self.assertTrue(torch.isclose(loss_result, loss_target))
