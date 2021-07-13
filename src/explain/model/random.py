import torch


class RandomExplanation:
    def __init__(self, model, args):
        self.args = args
        self.explain_config = args.explain_config
        self.model = model 

    def __call__(self, batch, **kwargs):
        attention_mask = batch['attention_mask']
        scores = torch.rand(*attention_mask.size())
        scores = scores * attention_mask
        scores = torch.nn.functional.softmax(scores, dim=-1)
        return scores # Tensor [B, L]
