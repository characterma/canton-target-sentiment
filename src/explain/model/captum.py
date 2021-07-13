import captum.attr as captum_attr # https://github.com/pytorch/captum/blob/master/captum/attr/__init__.py
import torch
from explain.model.base import NLPExplanation


class LayerIntegratedGradients(NLPExplanation):
    def __init__(self, model, args):
        super().__init__(model=model, args=args)
        model_class = getattr(captum_attr, "LayerIntegratedGradients")
        self.explain_model = model_class(
            self.get_logit, 
            self.model.pretrained_model.embeddings
        )

    def __call__(self, batch, **kwargs):
        """
        """
        # make inputs
        inputs = self.make_inputs(batch)

        target = kwargs.get('target', None)
        if target is None:
            target = self.get_prediction(**inputs)

        additional_forward_args = tuple([inputs[col] for col in inputs if col!="input_ids"])
        attributions = self.explain_model.attribute(
            inputs=inputs['input_ids'], 
            target=target, 
            additional_forward_args=additional_forward_args, 
            internal_batch_size=self.explain_config['batch_size']
        )
        scores = attributions.sum(dim=-1) 
        return scores



class Lime(NLPExplanation):
    def __init__(self, model, args):
        super().__init__(model=model, args=args)
        model_class = getattr(captum_attr, "Lime")
        self.explain_model = model_class(self.get_logit)

    def __call__(self, batch, **kwargs):
        """
        """
        # make inputs
        inputs = self.make_inputs(batch)

        target = kwargs.get('target', None)
        if target is None:
            target = self.get_prediction(**inputs)

        additional_forward_args = tuple([inputs[col] for col in inputs if col!="input_ids"])
        attributions = self.explain_model.attribute(
            inputs=inputs['input_ids'], 
            target=target, 
            additional_forward_args=additional_forward_args, 
            n_samples=100
        )
        return attributions
