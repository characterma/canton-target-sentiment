from captum.attr import (
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
    LayerIntegratedGradients
)
import torch


class CaptumExplanation:
    def __init__(self, model, args):
        self.args = args
        self.explain_config = args.explain_config
        self.model = model 
        explain_model_class = eval(args.explain_config['model_class'])
        self.explain_model = explain_model_class(
            self.get_logit, 
            self.model.pretrained_model.embeddings
        )

    def get_prediction(self, **inputs):
        outputs = self.model(**inputs)
        return outputs[1]

    def get_logit(self, *args):
        outputs = self.model(*args)
        return outputs[2]

    def __call__(self, batch, target=None):
        """
        batch: dict of tensor
        target: int or list or tensor [1,2,0,..]
        """
        # make inputs
        inputs = dict()
        # print(batch)
        for col in batch:
            if torch.is_tensor(batch[col]):
                inputs[col] = batch[col].to(self.args.device).long()

        if target is None:
            target = self.get_prediction(**inputs)

        additional_forward_args = tuple([inputs[col] for col in inputs if col!="input_ids"])
        attributions, delta = self.explain_model.attribute(
            inputs=inputs['input_ids'], 
            target=target, 
            additional_forward_args=additional_forward_args, 
            return_convergence_delta=True, 
            internal_batch_size=self.explain_config['batch_size']
        )
        scores = attributions.sum(dim=-1) 
        # remove [PAD] and special tokens # tha (0.1) #nk (0.5)
        return scores # [B, L]
