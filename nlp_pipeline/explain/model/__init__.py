import torch
import torch.nn.functional as F
import captum.attr as captum_attr
from captum._utils.models.linear_model import SkLearnLasso
from captum._utils.gradient import compute_layer_gradients_and_eval
from captum._utils.common import _format_additional_forward_args, _format_input

from nlp_pipeline.explain.model.word_omission import WordOmission


class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        return outputs['logits']


class ExplainModel:
    def __init__(self, model, config):
        # self.args = args
        self.model = model
        self.config = config 

        self.method = self.config["method"]
        self.layer = self.config.get("layer", None)
        self.norm = self.config.get("norm", None)

        self.wrapped_model = WrappedModel(model=model)
        self.init_explain_model()

    def similarity_func(self, original_inp, perturbed_inp, interpretable_inp, **kwargs):
        original_am = original_inp[1]
        perturbed_am = perturbed_inp[1]
        ori_len = original_am.sum(dim=-1)
        new_len = perturbed_am.sum(dim=-1)
        dist = (ori_len - new_len).float() / ori_len
        sim = 1 - torch.exp(dist)
        return sim

    def bernoulli_perturb(self, inputs, **kwargs):
        attention_mask = inputs[1]
        probs = torch.ones_like(attention_mask) * 0.5
        perturb = torch.bernoulli(probs).long()
        perturb[attention_mask!=1] = 0
        perturb[:, 0] = 1
        perturb[:, attention_mask.sum().item() - 1] = 1
        return perturb

    def interp_to_input(self, interp_sample, inputs, **kwargs):
        input_ids = inputs[0]
        attention_mask = inputs[1]
        max_length = input_ids.shape[-1]
        input_ids = input_ids[interp_sample==1]
        attention_mask = attention_mask[interp_sample==1]
        cur_length = input_ids.shape[-1]
        input_ids = F.pad(input_ids, (0, max_length - cur_length), "constant", 0)
        attention_mask = F.pad(attention_mask, (0, max_length - cur_length), "constant", 0)
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
        return (input_ids, attention_mask)

    def get_default_target(self, inputs):
        logits = self.wrapped_model(**inputs)
        target = logits.argmax(-1).item()
        probs = F.softmax(logits, dim=-1)
        prob = probs[0, target].item()
        return target, prob

    def summarize_attributions(self, x):
        if self.norm is None:
            return x.mean(dim=-1) 
        elif self.norm == "l2":
            return x.norm(p=2, dim=-1)
        elif self.norm == "sum":
            x = x.sum(dim=-1).squeeze(0)
            x = x / torch.norm(x)
            return x.unsqueeze(0)

    def init_explain_model(self):
        if self.method == "Random":
            self.explain_model = None
        elif self.method == "Saliency":
            pass

        elif self.method == "GradientXActivation":
            if self.layer:
                layer = eval("self.model." + self.layer)
                self.explain_model = captum_attr.LayerGradientXActivation(
                    forward_func=self.wrapped_model, 
                    layer=layer
                )
            else:
                self.explain_model = captum_attr.GradientXActivation(
                    forward_func=self.wrapped_model
                )

        elif self.method == "IntegratedGradients":
            if self.layer:
                layer = eval("self.model." + self.layer)
                self.explain_model = captum_attr.LayerIntegratedGradients(
                    forward_func=self.wrapped_model, 
                    layer=layer
                )
            else:
                self.explain_model = captum_attr.IntegratedGradients(
                    forward_func=self.wrapped_model
                )

        elif self.method == "DeepLift":
            if self.layer:
                layer = eval("self.model." + self.layer)
                self.explain_model = captum_attr.LayerDeepLift(
                    model=self.wrapped_model,
                    layer=layer
                ) 
            else:
                self.explain_model = captum_attr.DeepLift(
                    forward_func=self.wrapped_model
                )

        elif self.method == "Lime":
            self.explain_model = captum_attr.LimeBase(
                forward_func=self.wrapped_model,
                interpretable_model=SkLearnLasso(alpha=0.08),
                similarity_func=self.similarity_func,
                perturb_func=self.bernoulli_perturb,
                perturb_interpretable_space=True,
                from_interp_rep_transform=self.interp_to_input,
                to_interp_rep_transform=None
            )

        elif self.method == "WordOmission":
            self.explain_model = WordOmission(
                model=self.wrapped_model,
            ) 

    def get_bert_ref_input_ids(self, inputs, pad_token_id, sep_token_id, cls_token_id):
        seq_len = inputs['attention_mask'].sum().item() - 2
        ref_input_ids = [cls_token_id] + [pad_token_id] * seq_len + [sep_token_id]
        return torch.tensor([ref_input_ids], device=inputs['attention_mask'].device)

    def __call__(self, inputs, target=None, **kwargs):
        target_prob = None

        if self.method == "Random":
            attention_mask = inputs["attention_mask"]
            scores = torch.rand(*attention_mask.size())
            scores = scores.to(attention_mask.device)
            scores = scores * attention_mask

        elif self.method == "Saliency":
            if target is None:
                target, target_prob = self.get_default_target(inputs)
            additional_forward_args = tuple(
                [inputs[col] for col in inputs if col != "input_ids"]
            )
            additional_forward_args = _format_additional_forward_args(
                additional_forward_args
            )
            layer = eval("self.model." + self.layer)
            layer_gradients, layer_evals = compute_layer_gradients_and_eval(
                self.wrapped_model,
                layer=layer,
                inputs=_format_input(inputs["input_ids"]),
                target_ind=target,
                additional_forward_args=additional_forward_args,
            )
            scores = self.summarize_attributions(layer_gradients[0])

        elif self.method == "GradientXActivation":
            additional_forward_args = tuple(
                [inputs[col] for col in inputs if col != "input_ids"]
            )
            if target is None:
                target, target_prob = self.get_default_target(inputs)
            attributions = self.explain_model.attribute(
                inputs=inputs["input_ids"],
                target=target,
                additional_forward_args=additional_forward_args,
            )
            scores = self.summarize_attributions(attributions)

        elif self.method == "IntegratedGradients":
            if  (kwargs.get('pad_token_id') is not None and 
            kwargs.get('sep_token_id') is not None and 
            kwargs.get('cls_token_id') is not None):
            
                baselines = self.get_bert_ref_input_ids(
                    inputs=inputs, 
                    pad_token_id=kwargs.get('pad_token_id'), 
                    sep_token_id=kwargs.get('sep_token_id'), 
                    cls_token_id=kwargs.get('cls_token_id')
                )
            else:
                baselines = None

            additional_forward_args = tuple(
                [inputs[col] for col in inputs if col != "input_ids"]
            )
            if target is None:
                target, target_prob = self.get_default_target(inputs)
            attributions = self.explain_model.attribute(
                inputs=inputs["input_ids"],
                target=target,
                baselines=baselines,
                additional_forward_args=additional_forward_args,
            )
            scores = self.summarize_attributions(attributions)

        elif self.method == "DeepLift":
            additional_forward_args = tuple(
                [inputs[col] for col in inputs if col != "input_ids"]
            )
            if target is None:
                target, target_prob = self.get_default_target(inputs)
            attributions = self.explain_model.attribute(
                inputs=inputs["input_ids"],
                target=target,
                additional_forward_args=additional_forward_args,
            )
            scores = self.summarize_attributions(attributions)

        elif self.method == "Lime":
            n_samples = inputs.get("n_samples", 500)
            if target is None:
                target, target_prob = self.get_default_target(inputs)
            attributions = self.explain_model.attribute(
                inputs=(inputs["input_ids"], inputs['attention_mask']),
                additional_forward_args=(None,),
                target=target,
                n_samples=500, 
                show_progress=True
            )
            scores = attributions

        elif self.method == "WordOmission":
            if target is None:
                target, target_prob = self.get_default_target(inputs)
            scores = self.explain_model.attribute(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                target=target.item(),
            )
        else:
            scores = None
        return scores, target, target_prob
