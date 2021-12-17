import torch
import torch.nn as nn
from transformers import BertPreTrainedModel

from model.layer.fc import LinearLayer
from model.utils import load_pretrained_bert, load_pretrained_config, NLPModelOutput


class BERT_AVG(BertPreTrainedModel):
    def __init__(self, args):
        super(BERT_AVG, self).__init__(load_pretrained_config(args))
        self.model_config = args.model_config
        self.pretrained_model = load_pretrained_bert(args)

        hidden_size = self.pretrained_model.config.hidden_size
        output_hidden_dim = args.model_config['output_hidden_dim']
        output_hidden_act_func = args.model_config['output_hidden_act_func']

        self.num_labels = len(args.label_to_id)

        if output_hidden_dim is not None:
            h_dim = [output_hidden_dim, self.num_labels]
        else:
            h_dim = [self.num_labels]

        self.linear = LinearLayer(
            in_dim=hidden_size,
            h_dim=h_dim,
            activation=output_hidden_act_func,
            use_bn=False
        )

        self.loss_func = nn.CrossEntropyLoss(reduction="mean")
        self.to(args.device)

    def avg_pool(sel, h, attention_mask):
        attention_mask = attention_mask.unsqueeze(-1)
        h = h.masked_fill(attention_mask==0, float(0))
        h = h.sum(dim=1) / attention_mask.sum(dim=1)        
        return h

    def forward(
        self,
        input_ids,
        attention_mask,
        label=None,
    ):
        outputs = dict()
        lm = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=None,
            output_attentions=True,
            return_dict=True,
        )
        h = lm["last_hidden_state"]
        h = self.avg_pool(h, attention_mask=attention_mask)
        logits = self.linear(h)
        prediction = torch.argmax(logits, dim=1)
        if label is not None:
            loss = self.loss_func(
                logits,  # [N, C]
                label  # [N]
            )
        else:
            loss = None
        outputs = NLPModelOutput(
            loss=loss, 
            prediction=prediction, 
            logits=logits
        )
        return outputs
