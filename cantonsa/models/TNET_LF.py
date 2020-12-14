import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from cantonsa.layers.dynamic_rnn import DynamicLSTM
from cantonsa.models.base import BaseModel


class Absolute_Position_Embedding(nn.Module):
    def __init__(self, size=None, mode="sum", device=None):
        self.device = device
        self.size = size  # 必须为偶数
        self.mode = mode
        super(Absolute_Position_Embedding, self).__init__()

    def forward(self, x, pos_inx):
        if (self.size is None) or (self.mode == "sum"):
            self.size = int(x.size(-1))
        batch_size, seq_len = x.size()[0], x.size()[1]
        weight = self.weight_matrix(pos_inx, batch_size, seq_len).to(self.device)
        x = weight.unsqueeze(2) * x
        return x

    def weight_matrix(self, pos_inx, batch_size, seq_len):
        pos_inx = pos_inx.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(pos_inx[i][1]):
                relative_pos = pos_inx[i][1] - j
                weight[i].append(1 - relative_pos / 40)
            for j in range(pos_inx[i][1], seq_len):
                relative_pos = j - pos_inx[i][0]
                weight[i].append(1 - relative_pos / 40)
        weight = torch.tensor(weight)
        return weight


class TNET_LF(nn.Module):
    def __init__(
        self,
        model_config,
        num_labels,
        pretrained_emb=None,
        num_emb=None,
        pretrained_lm=None,
        device="cpu",
    ):
        super(TNET_LF, self).__init__()
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

        self.position = Absolute_Position_Embedding(device=device)
        self.lstm1 = DynamicLSTM(
            emb_dim,
            model_config["hidden_dim"],
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm2 = DynamicLSTM(
            embed_dim,
            model_config["hidden_dim"],
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.convs3 = nn.Conv1d(2 * model_config["hidden_dim"], 50, 3, padding=1)
        self.fc1 = nn.Linear(
            4 * model_config["hidden_dim"], 2 * model_config["hidden_dim"]
        )
        self.fc = nn.Linear(50, num_labels)
        self.to(device)

    def forward(self, raw_text, target, target_span, label=None):
        feature_len = torch.sum(raw_text != 0, dim=-1)
        aspect_len = torch.sum(target != 0, dim=-1)
        feature = self.embed(raw_text)
        aspect = self.embed(target)
        v, (_, _) = self.lstm1(feature, feature_len)
        e, (_, _) = self.lstm2(aspect, aspect_len)
        v = v.transpose(1, 2)
        e = e.transpose(1, 2)
        for i in range(2):
            a = torch.bmm(e.transpose(1, 2), v)
            a = F.softmax(a, 1)  # (aspect_len,context_len)
            aspect_mid = torch.bmm(e, a)
            aspect_mid = torch.cat((aspect_mid, v), dim=1).transpose(1, 2)
            aspect_mid = F.relu(self.fc1(aspect_mid).transpose(1, 2))
            v = aspect_mid + v
            v = self.position(v.transpose(1, 2), target_span).transpose(1, 2)
        z = F.relu(self.convs3(v))  # [(N,Co,L), ...]*len(Ks)
        z = F.max_pool1d(z, z.size(2)).squeeze(2)
        logits = self.fc(z)
        # Softmax
        if label is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), label.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), label.view(-1))
        return loss, logits
