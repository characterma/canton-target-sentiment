import torch
import torch.nn as nn
from transformers import BertPreTrainedModel

from model.layer.fc import FCLayer
from model.utils import load_pretrained_bert, load_pretrained_config


class BERT_CLS(BertPreTrainedModel):

    def __init__(self, args):
        super(BERT_CLS, self).__init__(
            load_pretrained_config(args.model_config['pretrained_lm'])
        )
        self.model_config = args.model_config
        self.pretrained_model = load_pretrained_bert(
            self.model_config['pretrained_lm']
        )

        hidden_size = self.pretrained_model.config.hidden_size
        self.num_labels = len(args.label_to_id)
        self.classifier = FCLayer(
            input_dim=hidden_size, 
            output_dim=self.num_labels, 
            dropout_rate=self.model_config["dropout_rate"],
        )
        self.loss_func = nn.CrossEntropyLoss(reduction="mean")
        self.to(args.device)
        
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
        h = h[:, 0, :]
        logits = self.classifier(h)
        prediction = torch.argmax(logits, dim=1).cpu().tolist()

        if label is not None:
            loss = self.loss_func(
                logits.view(-1, self.num_labels), # [N, C]
                label.view(-1) # [N]
            )
        else:
            loss = None

        return [loss, prediction, logits]
