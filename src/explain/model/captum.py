import captum.attr as captum_attr # https://github.com/pytorch/captum/blob/master/captum/attr/__init__.py
import torch
from explain.model.base import NLPExplanation


class LayerIntegratedGradients(NLPExplanation):
    def __init__(self, model, args):
        super().__init__(model=model, args=args)
        layer = eval("self.model." + args.explain_config['layer'])
        self.explain_model = captum_attr.LayerIntegratedGradients(
            forward_func=self.get_logit, 
            layer=layer
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


class LayerActivation(NLPExplanation):
    def __init__(self, model, args):
        super().__init__(model=model, args=args)
        layer = eval("self.model." + args.explain_config['layer'])
        self.explain_model = captum_attr.LayerActivation(
            forward_func=self.get_logit, 
            layer=layer
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
            # target=target, 
            additional_forward_args=additional_forward_args, 
        )
        scores = attributions.sum(dim=-1) 
        return scores


class LayerConductance(NLPExplanation):
    def __init__(self, model, args):
        super().__init__(model=model, args=args)
        layer = eval("self.model." + args.explain_config['layer'])
        self.explain_model = captum_attr.LayerConductance(
            forward_func=self.get_logit, 
            layer=layer
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


class InternalInfluence(NLPExplanation):
    def __init__(self, model, args):
        super().__init__(model=model, args=args)
        layer = eval("self.model." + args.explain_config['layer'])
        self.explain_model = captum_attr.InternalInfluence(
            forward_func=self.get_logit, 
            layer=layer
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


class LayerGradCam(NLPExplanation):
    def __init__(self, model, args):
        super().__init__(model=model, args=args)
        layer = eval("self.model." + args.explain_config['layer'])
        self.explain_model = captum_attr.LayerGradCam(
            forward_func=self.get_logit, 
            layer=layer
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
        )
        scores = attributions.sum(dim=-1) 
        return scores


class LayerDeepLift(NLPExplanation):
    def __init__(self, model, args):
        super().__init__(model=model, args=args)
        layer = eval("self.wrapped_model.model." + args.explain_config['layer'])
        self.explain_model = captum_attr.LayerDeepLift(
            model=self.wrapped_model, 
            layer=layer, 
        )

    def __call__(self, batch, **kwargs):
        """
        """
        # make inputs
        inputs = self.make_inputs(batch)

        target = kwargs.get('target', None)
        if target is None:
            target = self.get_prediction(**inputs)
        additional_forward_args = [inputs[col] for col in inputs if col!="input_ids"]
        # additional_forward_args = tuple([inputs[col] for col in inputs if col!="input_ids"])
        attributions = self.explain_model.attribute(
            inputs=inputs['input_ids'], 
            target=target, 
            additional_forward_args=inputs['attention_mask'], 
            # internal_batch_size=self.explain_config['batch_size']
        )
        scores = attributions.sum(dim=-1) 
        return scores


class LayerDeepLiftShap(NLPExplanation):
    def __init__(self, model, args):
        super().__init__(model=model, args=args)
        layer = eval("self.model." + args.explain_config['layer'])
        self.explain_model = captum_attr.LayerDeepLiftShap(
            model=self.get_logit, 
            layer=layer
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


class LayerGradientShap(NLPExplanation):
    def __init__(self, model, args):
        super().__init__(model=model, args=args)
        layer = eval("self.model." + args.explain_config['layer'])
        self.explain_model = captum_attr.LayerGradientShap(
            forward_func=self.get_logit, 
            layer=layer
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
            baselines=torch.zeros_like(inputs['input_ids']), 
            additional_forward_args=additional_forward_args, 
        )
        scores = attributions.sum(dim=-1) 
        return scores


class LayerLRP(NLPExplanation):
    def __init__(self, model, args):
        super().__init__(model=model, args=args)
        layer = eval("self.model." + args.explain_config['layer'])
        self.explain_model = captum_attr.LayerLRP(
            forward_func=self.get_logit, 
            layer=layer, 
            # multiply_by_inputs=False
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
        )
        scores = attributions.sum(dim=-1) 
        return scores


class LayerGradientXActivation(NLPExplanation):
    def __init__(self, model, args):
        super().__init__(model=model, args=args)
        layer = eval("self.model." + args.explain_config['layer'])
        self.explain_model = captum_attr.LayerGradientXActivation(
            forward_func=self.get_logit, 
            layer=layer, 
            # multiply_by_inputs=False
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
        )
        scores = attributions.sum(dim=-1) 
        return scores


class Lime(NLPExplanation):
    def __init__(self, model, args):
        super().__init__(model=model, args=args)
        self.explain_model = captum_attr.Lime(self.get_logit)

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
