import torch
import torch.nn as nn

from transformers.modeling_bert import BertEmbeddings
from model.layer.fc import FCLayer
from model.utils import load_pretrained_config


class TGSAN2(nn.Module):
    def __init__(self, args):
        super(TGSAN2, self).__init__()

        self.bert_config = load_pretrained_config(args.model_config)
        self.bert_config.hidden_size = args.model_config["emb_dim"]
        self.bert_config.hidden_dropout_prob = args.model_config["emb_dropout"]
        if hasattr(args, "vocab_size"):
            self.bert_config.vocab_size = args.vocab_size
        else:
            self.bert_config.vocab_size = self.bert_config.vocab_size
        self.bert_config.max_position_embeddings = args.model_config["max_length"]

        self.embedding = BertEmbeddings(config=self.bert_config)
        self.num_labels = len(args.label_to_id)

        self.encoders = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.bert_config.hidden_size,
                    nhead=12,
                    dim_feedforward=360,
                    dropout=args.model_config["encoder_dropout"],
                    activation=args.model_config.get("activation", "relu"),
                )
                for i in range(args.model_config["n_encoder"])
            ]
        )

        self.classifier = FCLayer(
            input_dim=self.bert_config.hidden_size,
            output_dim=self.num_labels,
            dropout_rate=args.model_config["fc_dropout"],
            activation=args.model_config.get("activation", None),
        )
        self.loss_fct = nn.CrossEntropyLoss(reduction="mean")
        self.to(args.device)

    def pool_target(self, hidden_output, t_mask):
        t_h = torch.max(
            hidden_output.float() * torch.unsqueeze(t_mask.float(), -1),
            dim=1,
            keepdim=False,
        )
        return t_h.values

    def forward(self, input_ids, attention_mask, target_mask, label=None, **kwargs):

        x = self.embedding(
            input_ids=input_ids,
            token_type_ids=target_mask,
            position_ids=None,
            inputs_embeds=None,
        )

        x = x.transpose(1, 0)
        for encoder in self.encoders:
            x = encoder(x, src_key_padding_mask=(1 - attention_mask).bool())  # B, L, D
        x = x.transpose(1, 0)

        tgt = self.pool_target(x, target_mask)
        logits = self.classifier(tgt)
        prediction = torch.argmax(logits, dim=1).cpu().tolist()

        if label is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), label.view(-1))
        else:
            loss = None

        return [loss, prediction, logits]
