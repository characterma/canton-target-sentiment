import torch
import torch.nn as nn
import numpy as np
from transformers import BertPreTrainedModel
import torch.nn.functional as F
from model.layer.fc import LinearLayer
from model.utils import load_pretrained_bert, load_pretrained_config


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.w = torch.nn.Linear(hidden_size, hidden_size, bias=False)  # [D,D]
        self.q = torch.nn.Parameter(torch.rand(hidden_size))

    def forward(self, h, mask):
        x = self.w(h)
        x = torch.matmul(x, self.q)
        x = x.masked_fill_(mask == 0, -float('Inf'))
        a = F.softmax(x, -1)
        return a


class ScaledDotProductAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.q = torch.nn.Parameter(torch.rand(hidden_size))
        self.sqrt_dim = np.sqrt(hidden_size)

    def forward(self, h, mask):
        x = torch.matmul(h, self.q) / self.sqrt_dim
        x = x.masked_fill_(mask == 0, -float('Inf'))
        a = F.softmax(x, -1)
        return a


class BERT_ATTN(BertPreTrainedModel):
    def __init__(self, args):
        super().__init__(load_pretrained_config(args.model_config))
        self.model_config = args.model_config
        self.pretrained_model = load_pretrained_bert(args)

        hidden_size = self.pretrained_model.config.hidden_size
        output_hidden_dim = args.model_config.get("output_hidden_dim", None)
        output_hidden_act_func = args.model_config.get("output_hidden_act_func", None)
        attention_type = args.model_config.get("attention_type", None)

        # reference to http://ess-repos01.wisers.com:9980/UAP/AI/common-dl-model/blob/master/net/layers/attention_layers.py
        # attention params
        if attention_type == "additive":
            self.attention = AdditiveAttention(hidden_size)
        elif attention_type == "dot-product":
            self.attention = ScaledDotProductAttention(hidden_size)
        else:
            self.attention = None 
            
        self.num_labels = len(args.label_to_id)

        # classifier params
        if output_hidden_dim is not None:
            h_dim = [output_hidden_dim, self.num_labels]
        else:
            h_dim = [self.num_labels]

        self.linear = LinearLayer(
            in_dim=hidden_size,
            h_dim=h_dim,
            activation=output_hidden_act_func,
            use_bn=False,
        )
        self.loss_func = nn.CrossEntropyLoss(reduction="mean")
        self.to(args.device)

    def forward(self, input_ids, attention_mask, label=None):
        outputs = dict()
        lm = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=None,
            return_dict=True,
        )
        h = lm["last_hidden_state"]  # [B,L,h_dim]

        a = self.attention(h, attention_mask)
        h = torch.bmm(a.unsqueeze(1), h)  # [*, 1, L] x [*, L, D]
        h = h.squeeze(1)

        logits = self.linear(h)
        prediction = torch.argmax(logits, dim=1).cpu().tolist()

        if label is not None:
            loss = self.loss_func(
                logits.view(-1, self.num_labels), label.view(-1)  # [N, C], [N]
            )
        else:
            loss = None

        outputs['loss'] = loss 
        outputs['prediction'] = prediction
        outputs['logits'] = logits
        outputs['attentions'] = a
        return outputs
