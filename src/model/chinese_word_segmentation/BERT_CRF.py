from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel

from model.layer.crf import LinearChainCRF
from model.layer.cnn import ConvLayer
from model.layer.fc import FCLayer
from model.utils import load_pretrained_bert, load_pretrained_config


class BERT_CRF(BertPreTrainedModel):

    def __init__(self, args):
        pretrained_config = load_pretrained_config(
            args.model_config['pretrained_lm']
        )
        super(BERT_CRF, self).__init__(pretrained_config)

        self.model_config = args.model_config
        self.pretrained_model = load_pretrained_bert(
            self.model_config['pretrained_lm']
        )
        if not args.model_config["embedding_trainable"]:
            self.freeze_emb()
        self.pretrained_config = pretrained_config
        self.num_labels = len(args.label_to_id)

        self.bert_dropout = nn.Dropout(self.model_config['bert_dropout'])
        self.crf = LinearChainCRF(self.pretrained_config.hidden_size, self.num_labels)
        
        self.to(args.device)

    def freeze_emb(self):
        # Freeze all parameters except self attention parameters
        for param_name, param in self.pretrained_model.named_parameters():
            if "embeddings" in param_name:
                param.requires_grad = False

    def forward(self, text, attention_mask, label=None):
        lm = self.pretrained_model(
            input_ids=text,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = lm["last_hidden_state"]
        logits = self.bert_dropout(logits)
        prediction, scores = self.crf.viterbi_decode(logits, length_index=attention_mask) # [B, 1, L], [B, 1]

        if label is not None:
            loss = self.crf.nll_loss(x=logits, y=label, length_index=attention_mask, reduce='mean')
        else:
            loss = None
        
        prediction = [p[0] for p in prediction]
        return [loss, prediction, logits] # # [[0], [Batch size, Sequence Length], [Batch size, Sequence Length, Label Number]]

