import math
import torch
import torch.nn as nn

from transformers.modeling_bert import BertEmbeddings
from model.layer.fc import FCLayer
from model.utils import load_pretrained_config


class SEQ2(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.bert_config = load_pretrained_config(args.model_config)
        self.bert_config.hidden_size = args.model_config["emb_dim"]
        self.bert_config.hidden_dropout_prob = args.model_config["emb_dropout"]
        if hasattr(args, 'vocab_size'):
            self.bert_config.vocab_size = args.vocab_size
        else:
            self.bert_config.vocab_size = self.bert_config.vocab_size
        self.bert_config.max_position_embeddings = args.model_config['max_length']

        self.embedding = BertEmbeddings(config=self.bert_config)
        self.num_labels = len(args.label_to_id)
        self.sequence_len = args.model_config['max_length']

        self.encoders = nn.ModuleList([nn.TransformerEncoderLayer(
            d_model=self.bert_config.hidden_size, 
            nhead=12, 
            dim_feedforward=360, 
            dropout=args.model_config['encoder_dropout'], 
            activation=args.model_config.get('activation', 'relu')
        ) for i in range(args.model_config['n_encoder'])])

        self.self_attn = nn.MultiheadAttention(self.bert_config.hidden_size, 1)

        self.fc1 = FCLayer(
            input_dim=self.bert_config.hidden_size,
            output_dim=self.num_labels,
            dropout_rate=args.model_config['fc_dropout'],
            activation=args.model_config.get("activation", None),
        )

        self.fc2 = FCLayer(
            input_dim=self.bert_config.hidden_size,
            output_dim=1,
            dropout_rate=args.model_config['fc_dropout'],
            activation="ReLU",
        )
        self.softmax = nn.Softmax(dim=-1)

        self.loss_fct = nn.CrossEntropyLoss(reduction="mean")
        self.to(args.device)

    def encode_tokens(self, input_ids):
        x = self.embedding(
            input_ids=input_ids, 
            position_ids=None, 
            inputs_embeds=None, 
        )
        x = x.transpose(1, 0)
        for encoder in self.encoders:
            x = encoder(x, src_key_padding_mask=(1-attention_mask).bool()) # B, L, D
        x = x.transpose(1, 0)
        return x 

    def forward(self, input_ids, attention_mask, label=None, label_e=None, **kwargs):
        x = self.encode_tokens(input_ids=input_ids)

        h = torch.mean(x.float(), dim=1, keepdim=False)
        logits = self.fc1(h)

        expl = self.fc2(x)
        expl = self.softmax(expl)

        if label is not None:
            loss = self.loss_func_1(logits, label)
        elif label_e is not None:
            loss = self.loss_func_2(expl, label_e)

        prediction = torch.argmax(logits, dim=1).cpu().tolist()
        return [loss, prediction, logits, explanation]
