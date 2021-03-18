import torch
import math
import torch.nn as nn

import transformers
from transformers.modeling_bert import BertEmbeddings
from .base import BaseModel

class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class TGSAN2(BaseModel):
    """Architecture:
        Embedding + Memory Builder + SCU + CFU + OUTPUT Layer
    Arguments:
        text {tensor} -- in shape [B, L]; tokenized input sequence, L is the length of the sequence
        tgt_idx {tensor} -- in shape [B, L]; target index of the input sequence, 1 for target tokens, 0 for others
        length_idx {tensor} -- in shape [B, L]; length index of the input sequence, 0 for padding tokens, 1 for others
    """

    INPUT_COLS = ["raw_text", "attention_mask", "target_mask", "label", "soft_label"]
    def __init__(
        self,
        model_config,
        num_labels,
        pretrained_emb=None,
        num_emb=None,
        pretrained_lm=None,
        bert_config=None, 
        device="cpu",
    ):
        super(TGSAN2, self).__init__()
        self.bert_config = bert_config
        self.num_labels = num_labels
        # print(bert_config)
        self.embedding = BertEmbeddings(config=bert_config)

        self.encoder = nn.TransformerEncoderLayer(
            d_model=bert_config.hidden_size, 
            nhead=12, 
            dim_feedforward=360, 
            dropout=0.1, 
            activation='relu'
        )

        self.classifier = FCLayer(
            input_dim=bert_config.hidden_size,
            output_dim=self.num_labels,
            dropout_rate=0.1,
            use_activation=False,
        )

        self.to(device)

    def pool_target(self, hidden_output, t_mask):
        t_h = torch.max(
            hidden_output.float() * torch.unsqueeze(t_mask.float(), -1),
            dim=1,
            keepdim=False,
        )
        return t_h.values


    def forward(self, raw_text, attention_mask, target_mask, label=None, soft_label=None):

        x = self.embedding(
            input_ids=raw_text, 
            token_type_ids=target_mask, 
            position_ids=None, 
            inputs_embeds=None, 
        )

        x = self.encoder(
            x.transpose(1, 0), 
            src_key_padding_mask=(1-attention_mask).bool()
        ).transpose(1, 0) # B, L, D

        tgt = self.pool_target(x, target_mask)
        logits = self.classifier(tgt)

        # print(tgt[torch.isnan(tgt)])

        if soft_label is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.view(-1), soft_label.view(-1))            
        elif label is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), label.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), label.view(-1))
        # print("***********************")

        # print(logits[torch.isnan(logits)])

        # print("***********************")
        # print(loss.grad)

        return loss, logits
