import abc
import torch


class NLPExplanation(abc.ABC):
    def __init__(self, model, args):
        self.args = args
        self.explain_config = args.explain_config
        self.model = model 
        self.explain_model = None

    def get_prediction(self, **inputs):
        outputs = self.model(**inputs)
        return outputs[1]

    def get_logit(self, *args):
        outputs = self.model(*args)
        return outputs[2]

    def make_inputs(self, batch):
        inputs = dict()
        for col in batch:
            if torch.is_tensor(batch[col]):
                inputs[col] = batch[col].to(self.args.device).long()
        return inputs

    @abc.abstractmethod
    def __call__(self, batch, **kwargs):
        """
        Args:
            batch: dict of tensor
        Return:
            scores: [B,L]
        """
        return NotImplemented