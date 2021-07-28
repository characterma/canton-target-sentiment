import torch
import captum.attr as captum_attr


class WrappedModel(torch.nn.Module):
    def __init__(self, model, logits_index=None):
        super().__init__()
        self.model = model
        self.logits_index = logits_index

    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        if self.logits_index is None:
            return outputs
        else:
            return outputs[self.logits_index]


class ExplainModel:
    def __init__(self, method, model, layer, logits_index=None):
        # self.args = args
        self.method = method
        self.model = model
        self.layer = layer
        self.logits_index = logits_index

        self.wrapped_model = WrappedModel(model=model, logits_index=logits_index)
        self.init_explain_model()

    def init_explain_model(self):
        if self.method == "Random":
            self.explain_model = None
        elif self.method == "LayerIntegratedGradients":
            layer = eval("self.model." + self.layer)
            self.explain_model = captum_attr.LayerIntegratedGradients(
                forward_func=self.wrapped_model, layer=layer
            )
        elif self.method == "LayerGradientXActivation":
            layer = eval("self.model." + self.layer)
            self.explain_model = captum_attr.LayerIntegratedGradients(
                forward_func=self.wrapped_model, layer=layer
            )
        # else:
        #     raise (ValueError)

    def __call__(self, inputs):
        if self.method == "Random":
            attention_mask = inputs["attention_mask"]
            scores = torch.rand(*attention_mask.size())
            scores = scores.to(attention_mask.device)
            scores = scores * attention_mask
            return scores

        elif self.method == "ModelOuput":
            outputs = self.model(**inputs)
            scores = outputs[3]
            return scores

        elif self.method == "LayerIntegratedGradients":
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
            return scores

        elif self.method == "LayerGradientXActivation":
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
            return scores
        else:
            return None
