import torch
import torch.nn as nn
from transformers import BertPreTrainedModel

from model.layer.fc import FCLayer, LinearLayer
from model.utils import load_pretrained_bert, load_pretrained_config


class BERT_AVG(BertPreTrainedModel):
    def __init__(self, args):
        super(BERT_AVG, self).__init__(load_pretrained_config(args.model_config))
        self.model_config = args.model_config
        self.pretrained_model = load_pretrained_bert(self.model_config)

        hidden_size = self.pretrained_model.config.hidden_size
        output_hidden_dim = args.model_config['output_hidden_dim']
        output_hidden_act_func = args.model_config['output_hidden_act_func']

        self.num_labels = len(args.label_to_id)

        self.linear = LinearLayer(
            in_dim=hidden_size, 
            h_dim=[output_hidden_dim, self.num_labels], 
            activation=output_hidden_act_func,
            use_bn=False
        )

        self.loss_func = nn.CrossEntropyLoss(reduction="mean")
        self.to(args.device)

    def avg_pool(sel, h):
        return torch.mean(
            h.float(),
            dim=1,
            keepdim=False,
        )

    def forward(
        self,
        input_ids,
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
        h = self.avg_pool(h)
        logits = self.linear(h)
        prediction = torch.argmax(logits, dim=1).cpu().tolist()

        if label is not None:
            loss = self.loss_func(
                logits.view(-1, self.num_labels), # [N, C]
                label.view(-1) # [N]
            )
        else:
            loss = None

        return [loss, prediction, logits]
