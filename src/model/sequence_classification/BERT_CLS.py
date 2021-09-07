import torch
import torch.nn as nn
from transformers import BertPreTrainedModel

from model.layer.fc import FCLayer, LinearLayer
from model.utils import load_pretrained_bert, load_pretrained_config


class BERT_CLS(BertPreTrainedModel):
    """
    https://huggingface.co/transformers/_modules/transformers/models/bert/modeling_bert.html#BertForSequenceClassification
    """
    def __init__(self, args):
        super(BERT_CLS, self).__init__(load_pretrained_config(args.model_config))
        self.model_config = args.model_config
        self.pretrained_model = load_pretrained_bert(args)
        hidden_size = self.pretrained_model.config.hidden_size
        self.num_labels = len(args.label_to_id)
        self.classifier = nn.Linear(hidden_size, self.num_labels)
        self.loss_func = nn.CrossEntropyLoss(reduction="mean")
        self.to(args.device)

    def forward(self, input_ids, attention_mask, label=None):
        outputs = dict()
        bert_outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=None,
            output_attentions=True,
            return_dict=True,
        )
        logits = self.classifier(bert_outputs[1])
        if label is not None:
            loss = self.loss_func(
                logits.view(-1, self.num_labels), label.view(-1)  # [N, C]  # [N]
            )
        else:
            loss = None
        prediction = torch.argmax(logits, dim=1).cpu().tolist()

        outputs['loss'] = loss
        outputs['prediction'] = prediction
        outputs['logits'] = logits
        outputs['attentions'] = bert_outputs['attentions']
        return outputs
