import torch
import math
import torch.nn as nn

import transformers
from transformers.modeling_bert import BertEmbeddings
from transformers import BertConfig
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
        device="cpu",
    ):
        super(TGSAN2, self).__init__()

        self.bert_config = BertConfig.from_pretrained(model_config["bert_config"])

        self.bert_config.hidden_size = model_config["emb_dim"]
        self.bert_config.hidden_dropout_prob = model_config["emb_dropout"]

        self.bert_config.vocab_size = num_emb
        self.bert_config.max_position_embeddings = model_config['max_length']


        self.num_labels = num_labels
        self.embedding = BertEmbeddings(config=self.bert_config)
        # self.embedding_dropout = nn.Dropout(model_config["emb_dropout"])

        self.encoders = nn.ModuleList([nn.TransformerEncoderLayer(
            d_model=self.bert_config.hidden_size, 
            nhead=12, 
            dim_feedforward=360, 
            dropout=model_config['encoder_dropout'], 
            # activation='relu'
        ) for i in range(model_config['n_encoder'])])

        self.classifier = FCLayer(
            input_dim=self.bert_config.hidden_size,
            output_dim=self.num_labels,
            dropout_rate=model_config['fc_dropout'],
            use_activation=False,
        )
        self.device = device
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

        # x = self.embedding_dropout(x)

        x = x.transpose(1, 0)
        for encoder in self.encoders:
            x = encoder(x, src_key_padding_mask=(1-attention_mask).bool()) # B, L, D
        x = x.transpose(1, 0)

        tgt = self.pool_target(x, target_mask)
        logits = self.classifier(tgt)

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
        else:
            loss = None
        # print("***********************")

        # print(logits[torch.isnan(logits)])

        # print("***********************")
        # print(loss.grad)

        return loss, logits
