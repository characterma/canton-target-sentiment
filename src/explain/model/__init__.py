import torch
import torch.nn.functional as F
import captum.attr as captum_attr
from captum._utils.models.linear_model import SkLearnLasso
from captum._utils.gradient import compute_layer_gradients_and_eval
from captum._utils.common import _format_additional_forward_args, _format_input
from explain.model.word_omission import WordOmission


class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        return outputs['logits']


class ExplainModel:
    def __init__(self,model, config):
        # self.args = args
        self.model = model
        self.config = config 

        self.method = self.config["method"]
        self.layer = self.config.get("layer", None)
        self.model_output = self.config.get("model_output", None)
        self.only_gradient = self.config.get("only_gradient", False)
        self.times_gradient = self.config.get("times_gradient", False)
        self.attn_agg_method = self.config.get("attn_agg_method", None)

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
        # print(perturb)
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
        # print(input_ids, attention_mask)
        return (input_ids, attention_mask)

    def init_explain_model(self):
        if self.method == "Random":
            self.explain_model = None
        elif self.method == "Saliency":
            pass
        elif self.method == "IntegratedGradients":
            if self.layer:
                layer = eval("self.model." + self.layer)
                self.explain_model = captum_attr.LayerIntegratedGradients(
                    forward_func=self.wrapped_model, layer=layer
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

    def __call__(self, inputs):
        if self.method == "Random":
            attention_mask = inputs["attention_mask"]
            scores = torch.rand(*attention_mask.size())
            scores = scores.to(attention_mask.device)
            scores = scores * attention_mask
        # such as attention
        elif self.method == "Custom":
            outputs = self.model(**inputs)
            if self.attn_agg_method is None:
                if not self.only_gradient:
                    scores = outputs[self.model_output]
                else:
                    scores = None
                    
            elif self.attn_agg_method == "ATTN_SUM":
                attns = outputs[self.model_output]  
                x = []
                for attn in list(attns):
                    attn = attn.sum(-2)  
                    attn = attn.mean(1)
                    x.append(attn)
                x = torch.stack(x, dim=0)
                scores = torch.mean(x, dim=0)

            elif self.attn_agg_method == "LABEL_ATTN":
                attns = outputs[self.model_output]  # [B, C, L] 
                target = inputs.get("target", None)
                if target is None:
                    logits = self.wrapped_model(**inputs)
                    target = logits.argmax(-1)
                scores = torch.index_select(attns, dim=1, index=target)
                scores = scores.squeeze(dim=1)

            elif self.attn_agg_method == "CLS_BACKPROP":
                attention_mask = inputs["attention_mask"]
                attns = outputs[self.model_output]  # Turple of [B, H, L, L]
                m = None 
                for a in attns:
                    a = a.mean(1)  # [B, H, S, S] -> [B, S, S]
                    a = a / (torch.sum(a, -1).unsqueeze(-1))  # [B, S, S]
                    if m is None:
                        m = a 
                    else:
                        m = torch.bmm(m, a)
                scores = m[:, 0, :]  # [CLS] token

            elif self.attn_agg_method == "ATTN_BACKPROP":
                attention_mask = inputs["attention_mask"]
                attns = outputs[self.model_output]  # Turple of [B, H, L, L]
                m = None 
                for a in attns:
                    a = a.mean(1)  # [B, H, S, S] -> [B, S, S]
                    a = a / (torch.sum(a, -1).unsqueeze(-1))  # [B, S, S]
                    if m is None:
                        m = a 
                    else:
                        m = torch.bmm(m, a)
                scores = m.mean(1)

            if self.times_gradient or self.only_gradient:
                target = inputs.get("target", None)
                if target is None:
                    logits = self.wrapped_model(**inputs)
                    target = logits.argmax(-1)
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
                if not self.only_gradient:
                    scores = scores * layer_gradients[0]
                else:
                    scores = layer_gradients[0]
        elif self.method == "Saliency":
            target = inputs.get("target", None)
            if target is None:
                logits = self.wrapped_model(**inputs)
                target = logits.argmax(-1)
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
            scores = layer_gradients[0].sum(dim=-1)  #  [1, 256, 768]
        elif self.method == "IntegratedGradients":
            additional_forward_args = tuple(
                [inputs[col] for col in inputs if col != "input_ids"]
            )
            target = inputs.get("target", None)
            if target is None:
                logits = self.wrapped_model(**inputs)
                target = logits.argmax(-1)
            attributions = self.explain_model.attribute(
                inputs=inputs["input_ids"],
                target=target,
                additional_forward_args=additional_forward_args,
            )
            scores = attributions.sum(dim=-1)
        elif self.method == "DeepLift":
            additional_forward_args = tuple(
                [inputs[col] for col in inputs if col != "input_ids"]
            )
            target = inputs.get("target", None)
            if target is None:
                logits = self.wrapped_model(**inputs)
                target = logits.argmax(-1)
            attributions = self.explain_model.attribute(
                inputs=inputs["input_ids"],
                target=target,
                additional_forward_args=additional_forward_args,
            )
            scores = attributions.sum(dim=-1)
        elif self.method == "Lime":
            target = inputs.get("target", None)
            n_samples = inputs.get("target", 500)
            if target is None:
                logits = self.wrapped_model(**inputs)
                target = logits.argmax(-1)
            attributions = self.explain_model.attribute(
                inputs=(inputs["input_ids"], inputs['attention_mask']),
                additional_forward_args=(None,),
                target=target,
                n_samples=500, 
                show_progress=True
            )
            scores = attributions
        elif self.method == "WordOmission":
            target = inputs.get("target", None)
            if target is None:
                logits = self.wrapped_model(**inputs)
                target = logits.argmax(-1)
            scores = self.explain_model.attribute(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                target=target.item(),
            )
        else:
            scores = None
        return scores
