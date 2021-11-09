import torch
import torch.nn as nn
from transformers import BertPreTrainedModel
import torch.nn.functional as F
from model.utils import load_pretrained_bert, load_pretrained_config
from model.layer.fc import LinearLayer


class TDBERT(BertPreTrainedModel):
    def __init__(self, args):
        pretrained_config = load_pretrained_config(args)
        super(TDBERT, self).__init__(pretrained_config)

        # assert target_pooling in ["mean", "max"]
        self.model_config = args.model_config
        self.pretrained_model = load_pretrained_bert(args)
        if not args.model_config["embedding_trainable"]:
            self.freeze_emb()

        self.pretrained_config = pretrained_config
        self.num_labels = len(args.label_to_id)

        output_hidden_dim = args.model_config["output_hidden_dim"]
        output_hidden_act_func = args.model_config["output_hidden_act_func"]

        if output_hidden_dim is not None:
            h_dim = [output_hidden_dim, self.num_labels]
        else:
            h_dim = [self.num_labels]

        self.linear = LinearLayer(
            in_dim=pretrained_config.hidden_size,
            h_dim=h_dim,
            activation=output_hidden_act_func,
            use_bn=False,
        )
        self.return_tensors = None
        self.loss_func = nn.CrossEntropyLoss(reduction="none")
        self.to(args.device)

    def pool_target(self, hidden_output, t_mask):
        """Pool the entity hidden state vectors (H_i ~ H_j)"""
        t_h = torch.max(
            hidden_output.float() * torch.unsqueeze(t_mask.float(), -1),
            dim=1,
            keepdim=False,
        )
        return t_h.values

    def freeze_emb(self):
        # Freeze all parameters except self attention parameters
        for param_name, param in self.pretrained_model.named_parameters():
            if "embeddings" in param_name:
                param.requires_grad = False

    def set_return_options(self, return_tensors=None):
        self.return_tensors = return_tensors

    def forward(
        self,
        input_ids,
        target_mask,
        attention_mask,
        token_type_ids,
        label=None,
        **kwargs
    ):
        lm = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        h = lm["last_hidden_state"]
        tgt_h = self.pool_target(
            h, target_mask
        )  # outputs: [B, S, Dim], target_mask: [B, S]

        logits = self.linear(tgt_h)

        if label is not None:
            loss = self.loss_func(logits.view(-1, self.num_labels), label.view(-1))
            loss = loss.mean()
        else:
            loss = None

        results = dict()
        results['loss'] = loss
        results['logits'] = logits
        results['prediction'] = torch.argmax(logits, dim=1)
        results['probabilities'] = F.softmax(logits, -1)

        if self.return_tensors is None:
            outputs = results
        else:
            outputs = [results[nme] for nme in self.return_tensors]
        return outputs
