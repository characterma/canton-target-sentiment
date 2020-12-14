import torch
import torch.nn as nn
from cantonsa.models.base import BaseModel
from cantonsa.layers.attention import Attention
from cantonsa.layers.squeeze_embedding import SqueezeEmbedding


class MEMNET(BaseModel):
    def locationed_memory(self, memory, memory_len):
        # here we just simply calculate the location vector in Model2's manner
        batch_size = memory.shape[0]
        seq_len = memory.shape[1]
        memory_len = memory_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for idx in range(memory_len[i]):
                weight[i].append(1 - float(idx + 1) / memory_len[i])
            for idx in range(memory_len[i], seq_len):
                weight[i].append(1)
        weight = torch.tensor(weight).to(self.opt.device)
        memory = weight.unsqueeze(2) * memory
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
        super(MEMNET, self).__init__()
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
        self.squeeze_embedding = SqueezeEmbedding(batch_first=True)
        self.attention = Attention(emb_dim, score_function="mlp")
        self.x_linear = nn.Linear(emb_dim, emb_dim)
        self.dense = nn.Linear(emb_dim, num_labels)
        self.device = device
        self.to(device)

    def forward(self, raw_text_without_target, target, label=None):

        memory_len = torch.sum(raw_text_without_target != 0, dim=-1)
        aspect_len = torch.sum(target != 0, dim=-1)
        nonzeros_aspect = torch.tensor(aspect_len, dtype=torch.float).to(self.device)

        memory = self.embed(raw_text_without_target)
        memory = self.squeeze_embedding(memory, memory_len)
        # memory = self.locationed_memory(memory, memory_len)
        aspect = self.embed(target)
        aspect = torch.sum(aspect, dim=1)
        aspect = torch.div(aspect, nonzeros_aspect.view(nonzeros_aspect.size(0), 1))
        x = aspect.unsqueeze(dim=1)
        for _ in range(self.model_config["hops"]):
            x = self.x_linear(x)
            out_at, _ = self.attention(memory, x)
            x = out_at + x
        x = x.view(x.size(0), -1)
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
