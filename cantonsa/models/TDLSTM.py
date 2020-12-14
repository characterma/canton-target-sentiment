import torch
import torch.nn as nn
from cantonsa.models.base import BaseModel
from cantonsa.layers.dynamic_rnn import DynamicLSTM


class TDLSTM(BaseModel):
    def __init__(
        self,
        model_config,
        num_labels,
        pretrained_emb=None,
        num_emb=None,
        pretrained_lm=None,
        device="cpu",
    ):
        super(TDLSTM, self).__init__()
        self.num_labels = num_labels
        if pretrained_emb is not None:
            # print("******", pretrained_emb.shape)
            _, emb_dim = pretrained_emb.shape
            self.embed = nn.Embedding.from_pretrained(
                torch.tensor(pretrained_emb),
                freeze=(not model_config["embedding_trainable"]),
            )
        else:
            emb_dim = model_config["emb_dim"]
            self.embed = nn.Embedding(num_embeddings=num_emb, embedding_dim=emb_dim)
        self.lstm_l = DynamicLSTM(
            emb_dim, model_config["hidden_dim"], num_layers=1, batch_first=True
        )
        self.lstm_r = DynamicLSTM(
            emb_dim, model_config["hidden_dim"], num_layers=1, batch_first=True
        )
        self.dense = nn.Linear(model_config["hidden_dim"] * 2, num_labels)
        self.softmax = nn.Softmax(dim=-1)
        self.to(device)

    def forward(self, target_left_inclu, target_right_inclu, label=None):
        # print(target_left_inclu)
        # print(target_right_inclu)
        x_l, x_r = target_left_inclu, target_right_inclu
        x_l_len, x_r_len = torch.sum(x_l != 0, dim=-1), torch.sum(x_r != 0, dim=-1)
        x_l, x_r = self.embed(x_l), self.embed(x_r)
        _, (h_n_l, _) = self.lstm_l(x_l, x_l_len)
        _, (h_n_r, _) = self.lstm_r(x_r, x_r_len)
        h_n = torch.cat((h_n_l[0], h_n_r[0]), dim=-1)
        logits = self.dense(h_n)
        # logits = self.softmax(logits)
        # Softmax
        if label is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), label.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), label.view(-1))
        return loss, logits
