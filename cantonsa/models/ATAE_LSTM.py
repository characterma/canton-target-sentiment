from cantonsa.layers.attention import Attention, NoQueryAttention
from cantonsa.layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn
from cantonsa.layers.squeeze_embedding import SqueezeEmbedding
from cantonsa.models.base import BaseModel


class ATAE_LSTM(BaseModel):
    def __init__(
        self,
        model_config,
        num_labels,
        pretrained_emb=None,
        num_emb=None,
        pretrained_lm=None,
        device="cpu",
    ):
        super(ATAE_LSTM, self).__init__()
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
        self.squeeze_embedding = SqueezeEmbedding()
        self.lstm = DynamicLSTM(
            emb_dim * 2, model_config["hidden_dim"], num_layers=1, batch_first=True
        )
        self.attention = NoQueryAttention(
            model_config["hidden_dim"] + emb_dim, score_function="bi_linear"
        )
        self.dense = nn.Linear(model_config["hidden_dim"], num_labels)
        self.device = device
        self.to(device)

    def forward(self, raw_text, target, label=None):

        x_len = torch.sum(raw_text != 0, dim=-1)
        x_len_max = torch.max(x_len)
        aspect_len = torch.tensor(
            torch.sum(raw_text != 0, dim=-1), dtype=torch.float
        ).to(self.device)

        x = self.embed(raw_text)
        x = self.squeeze_embedding(x, x_len)
        aspect = self.embed(raw_text)
        aspect_pool = torch.div(
            torch.sum(aspect, dim=1), aspect_len.view(aspect_len.size(0), 1)
        )
        aspect = torch.unsqueeze(aspect_pool, dim=1).expand(-1, x_len_max, -1)
        x = torch.cat((aspect, x), dim=-1)

        h, (_, _) = self.lstm(x, x_len)
        ha = torch.cat((h, aspect), dim=-1)
        _, score = self.attention(ha)
        output = torch.squeeze(torch.bmm(score, h), dim=1)

        logits = self.dense(output)

        # Softmax
        if label is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), label.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), label.view(-1))
        return loss, logits
