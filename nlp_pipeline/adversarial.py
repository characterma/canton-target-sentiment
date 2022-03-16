import torch


def get_adversarial_class(name):
    return eval(name)


class PGD():
    def __init__(self, model, param_names):
        self.model = model
        self.param_names = param_names
        self.param_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and any([p in name for p in self.param_names]):
                if is_first_attack:
                    self.param_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and any([p in name for p in self.param_names]): 
                assert name in self.param_backup
                param.data = self.param_backup[name]
        self.param_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.param_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.param_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]


class FGM():
    def __init__(self, model, param_names):
        self.model = model
        self.param_names = param_names
        self.backup = {}

    def attack(self, epsilon=1.):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and any([p in name for p in self.param_names]):
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and any([p in name for p in self.param_names]): 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}