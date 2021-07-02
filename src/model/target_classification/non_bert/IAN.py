from cantonsa.layers.dynamic_rnn import DynamicLSTM
from cantonsa.layers.attention import Attention
from cantonsa.models.base import BaseModel
import torch
import torch.nn as nn


class IAN(BaseModel):
    def __init__(
        self,
        model_config,
        num_labels,
        pretrained_emb=None,
        num_emb=None,
        pretrained_lm=None,
        device="cpu",
    ):
        super(IAN, self).__init__()

        self.model_config = model_config
        self.num_labels = num_labels
        if pretrained_emb is not None:
            _, emb_dim = pretrained_emb.shape
            self.embed = nn.Embedding.from_pretrained(
                torch.tensor(pretrained_emb),
                freeze=(not model_config["embedding_trainable"]),
            )
        else:
            emb_dim = model_config["emb_dim"]
            self.embed = nn.Embedding(num_embeddings=num_emb, embedding_dim=emb_dim)
        self.lstm_context = DynamicLSTM(
            emb_dim, model_config["hidden_dim"], num_layers=1, batch_first=True
        )
        self.lstm_aspect = DynamicLSTM(
            emb_dim, model_config["hidden_dim"], num_layers=1, batch_first=True
        )
        self.attention_aspect = Attention(
            model_config["hidden_dim"], score_function="bi_linear"
        )
        self.attention_context = Attention(
            model_config["hidden_dim"], score_function="bi_linear"
        )
        self.dense = nn.Linear(model_config["hidden_dim"] * 2, num_labels)

        self.device = device
        self.to(device)

    def forward(self, raw_text, target, label=None):
        text_raw_len = torch.sum(raw_text != 0, dim=-1)
        aspect_len = torch.sum(target != 0, dim=-1)

        context = self.embed(raw_text)
        aspect = self.embed(target)
        context, (_, _) = self.lstm_context(context, text_raw_len)
        aspect, (_, _) = self.lstm_aspect(aspect, aspect_len)

        aspect_len = torch.tensor(aspect_len, dtype=torch.float).to(self.device)
        aspect_pool = torch.sum(aspect, dim=1)
        aspect_pool = torch.div(aspect_pool, aspect_len.view(aspect_len.size(0), 1))

        text_raw_len = torch.tensor(text_raw_len, dtype=torch.float).to(self.device)
        context_pool = torch.sum(context, dim=1)
        context_pool = torch.div(
            context_pool, text_raw_len.view(text_raw_len.size(0), 1)
        )

        aspect_final, _ = self.attention_aspect(aspect, context_pool)
        aspect_final = aspect_final.squeeze(dim=1)
        context_final, _ = self.attention_context(context, aspect_pool)
        context_final = context_final.squeeze(dim=1)

        x = torch.cat((aspect_final, context_final), dim=-1)
        logits = self.dense(x)

        # Softmax
        if label is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), label.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), label.view(-1))
        return loss, logits
