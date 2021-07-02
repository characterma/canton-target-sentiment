import torch
import torch.nn as nn
import torch.nn.functional as F
from cantonsa.models.base import BaseModel
from cantonsa.layers.dynamic_rnn import DynamicLSTM


class RAM(BaseModel):
    def locationed_memory(self, memory, memory_len, left_len, aspect_len):
        batch_size = memory.shape[0]
        seq_len = memory.shape[1]
        memory_len = memory_len.cpu().numpy()
        left_len = left_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        u = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for idx in range(left_len[i]):
                weight[i].append(1 - (left_len[i] - idx) / memory_len[i])
                u[i].append(idx - left_len[i])
            for idx in range(left_len[i], left_len[i] + aspect_len[i]):
                weight[i].append(1)
                u[i].append(0)
            for idx in range(left_len[i] + aspect_len[i], memory_len[i]):
                weight[i].append(
                    1 - (idx - left_len[i] - aspect_len[i] + 1) / memory_len[i]
                )
                u[i].append(idx - left_len[i] - aspect_len[i] + 1)
            for idx in range(memory_len[i], seq_len):
                weight[i].append(1)
                u[i].append(0)
        u = torch.tensor(u, dtype=memory.dtype).to(self.device).unsqueeze(2)
        weight = torch.tensor(weight).to(self.device).unsqueeze(2)
        v = memory * weight
        memory = torch.cat([v, u], dim=2)
        return memory

    def __init__(
        self,
        model_config,
        num_labels,
        pretrained_emb=None,
        num_emb=None,
        pretrained_lm=None,
        device="cpu",
    ):
        super(RAM, self).__init__()
        self.model_config = model_config
        self.num_labels = num_labels
        self.hops = model_config["hops"]
        if pretrained_emb is not None:
            _, self.emb_dim = pretrained_emb.shape
            self.embed = nn.Embedding.from_pretrained(
                torch.tensor(pretrained_emb),
                freeze=(not model_config["embedding_trainable"]),
            )
        else:
            emb_dim = model_config["emb_dim"]
            self.embed = nn.Embedding(num_embeddings=num_emb, embedding_dim=emb_dim)
            self.emb_dim = emb_dim
        self.bi_lstm_context = DynamicLSTM(
            self.emb_dim,
            model_config["hidden_dim"],
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.att_linear = nn.Linear(
            model_config["hidden_dim"] * 2 + 1 + self.emb_dim * 2, 1
        )
        self.gru_cell = nn.GRUCell(model_config["hidden_dim"] * 2 + 1, self.emb_dim)
        self.dense = nn.Linear(self.emb_dim, num_labels)
        self.device = device
        self.to(device)

    def forward(self, raw_text, target, target_left, label=None):
        """"""
        left_len = torch.sum(target_left != 0, dim=-1)
        memory_len = torch.sum(raw_text != 0, dim=-1)
        aspect_len = torch.sum(target != 0, dim=-1)
        nonzeros_aspect = aspect_len.float()

        memory = self.embed(raw_text)
        memory, (_, _) = self.bi_lstm_context(memory, memory_len)
        memory = self.locationed_memory(memory, memory_len, left_len, aspect_len)

        aspect = self.embed(target)
        aspect = torch.sum(aspect, dim=1)
        aspect = torch.div(aspect, nonzeros_aspect.unsqueeze(-1))
        et = torch.zeros_like(aspect).to(self.device)

        batch_size = memory.size(0)
        seq_len = memory.size(1)
        for _ in range(self.hops):
            g = self.att_linear(
                torch.cat(
                    [
                        memory,
                        torch.zeros(batch_size, seq_len, self.embed_dim).to(self.device)
                        + et.unsqueeze(1),
                        torch.zeros(batch_size, seq_len, self.embed_dim).to(self.device)
                        + aspect.unsqueeze(1),
                    ],
                    dim=-1,
                )
            )
            alpha = F.softmax(g, dim=1)
            i = torch.bmm(alpha.transpose(1, 2), memory).squeeze(1)
            et = self.gru_cell(i, et)
        logits = self.dense(et)
        # Softmax
        if label is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), label.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), label.view(-1))
        return loss, logits
