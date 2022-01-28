import torch
import torch.nn as nn
from transformers import BertPreTrainedModel

from model.layer.fc import FCLayer, LinearLayer
from model.utils import load_pretrained_bert, load_pretrained_config, NLPModelOutput


class BERT_CLS(BertPreTrainedModel):
    """
    Reference to:
        https://huggingface.co/transformers/_modules/transformers/models/bert/modeling_bert.html#BertForSequenceClassification
    """
    def __init__(self, args):
        super(BERT_CLS, self).__init__(load_pretrained_config(args))
        self.model_config = args.model_config
        self.pretrained_model = load_pretrained_bert(args)
        self.num_labels = len(args.label_to_id)
        self.dropout = nn.Dropout(
            p=args.model_config["classifier_dropout"]
        )
        self.classifier = nn.Linear(
            in_features=self.pretrained_model.config.hidden_size, 
            out_features=self.num_labels
        )
        self.loss_func = nn.CrossEntropyLoss(reduction="mean")
        self.return_logits = False
        self.to(args.device)

    def set_return_logits(self):
        self.return_logits = True
        
    def forward(self, input_ids, attention_mask, label=None):
        outputs = dict()
        bert_outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=None,
            output_attentions=True,
            return_dict=True,
        )
        pooler_output = bert_outputs["pooler_output"]
        pooler_output = self.dropout(pooler_output)
        logits = self.classifier(pooler_output)
        
        if self.return_logits:
            return logits 
        else:
            if label is not None:
                loss = self.loss_func(
                    logits.view(-1, self.num_labels), label.view(-1)  # [N, C]  # [N]
                )
            else:
                loss = None
            prediction = torch.argmax(logits, dim=1)
            outputs = NLPModelOutput(
                loss=loss, 
                prediction=prediction, 
                logits=logits
            )
            return outputs
