import torch
import math
import torch.nn as nn

import transformers
from .base import BaseModel


class StructuredSelfAttention(nn.Module):
    """Formula:
    align(X) = W2*tanh(W1*X^T)
    Penalty-Term P = Frobenius(A*A^T - I)
    """

    def __init__(
        self, in_dim, h_dim, r=1, dropout=0.0, scaled_att=True, penal_coeff=0.0
    ):
        super(StructuredSelfAttention, self).__init__()
        # hyper-params
        self.scaled_att = scaled_att
        self.penal_coeff = penal_coeff
        if self.penal_coeff > 0.0:
            self.I = nn.Parameter(
                torch.eye(r, dtype=torch.float32), requires_grad=False
            )
        # layers
        self.w1 = nn.Linear(in_dim, h_dim, bias=False)
        self.w2 = nn.Linear(h_dim, r, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.init_weight()

    def init_weight(self):
        torch.nn.init.xavier_uniform_(self.w1.weight.data, gain=1.0)
        torch.nn.init.xavier_uniform_(self.w2.weight.data, gain=1.0)

    def forward(self, x, length_index):
        att_score = self.w2(torch.tanh(self.w1(x))).transpose(-2, -1)  # [B, r, L]
        if self.scaled_att:
            att_score = att_score * (x.size(-1) ** -0.5)
        if length_index is not None:
            mask = length_index.unsqueeze(1)
            att_score.masked_fill_(mask == 0, -1e9)

        att = torch.softmax(att_score, dim=-1)
        if self.dropout is not None:
            att = self.dropout(att)

        penal_term = self.calculate_penal_term(att) if self.penal_coeff > 0.0 else None
        return att.matmul(x), att, penal_term  # [B, r, H], [B, r, L]

    def calculate_penal_term(self, att):
        att_T = torch.transpose(att, -2, -1)
        p = torch.norm((torch.bmm(att, att_T) - self.I), p="fro", dim=(1, 2)).mean()
        return p * self.penal_coeff


class StructuredAttention(nn.Module):
    """Formula:
    align(Q, K) = Q*tanh(W*K)
    Penalty-Term P = Frobenius(A*A^T - I)
    """

    def __init__(
        self, q_dim, k_dim, q_len, dropout=0.0, scaled_att=True, penal_coeff=0.0
    ):
        super(StructuredAttention, self).__init__()
        # hyper-params
        self.scaled_att = scaled_att
        self.penal_coeff = penal_coeff
        if self.penal_coeff > 0.0:
            self.I = nn.Parameter(
                torch.eye(q_len, dtype=torch.float32), requires_grad=False
            )
        # layers
        self.w = nn.Linear(k_dim, q_dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.init_weight()

    def init_weight(self):
        torch.nn.init.xavier_uniform_(self.w.weight.data, gain=1.0)

    def forward(self, q, k, v, k_length_index):
        att_score = torch.bmm(
            q, torch.tanh(self.w(k)).transpose(-2, -1)
        )  # [B, q_L, k_L]
        if self.scaled_att:
            att_score = att_score * (k.size(-1) ** -0.5)
        if k_length_index is not None:
            mask = k_length_index.unsqueeze(1)
            att_score.masked_fill_(mask == 0, -1e9)

        att = torch.softmax(att_score, dim=-1)
        if self.dropout is not None:
            att = self.dropout(att)

        penal_term = self.calculate_penal_term(att) if self.penal_coeff > 0.0 else None
        return att.matmul(v), att, penal_term  # [B, q_L, H], [B, q_L, k_L]

    def calculate_penal_term(self, att):
        att_T = torch.transpose(att, -2, -1)
        p = torch.norm((torch.bmm(att, att_T) - self.I), p="fro", dim=(1, 2)).mean()
        return p * self.penal_coeff


class BilinearAttention(nn.Module):
    """Formula: align(q, K) = q^T * W * K"""

    def __init__(self, q_dim, k_dim, scaled_att=True, dropout=0.0):
        super(BilinearAttention, self).__init__()
        self.scaled_att = scaled_att
        self.w = nn.Parameter(torch.FloatTensor(q_dim, k_dim), requires_grad=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.w.data, gain=1.0)

    def forward(self, q, k, v, k_length_index):
        att_score = q.matmul(self.w).matmul(k.transpose(-2, -1))
        if self.scaled_att:
            att_score = att_score * (k.size(-1) ** -0.5)
        if k_length_index is not None:
            mask = k_length_index.unsqueeze(1)
            for _ in range(len(k.size()) - 3):
                mask.unsqueeze_(1)
            att_score.masked_fill_(mask == 0, -1e9)

        att = torch.softmax(att_score, dim=-1)
        if self.dropout is not None:
            att = self.dropout(att)
        return torch.matmul(att, v), att  # [B, 1, H] [B, 1, k_L]


class PositionWiseFeedForwardLayer(nn.Module):
    def __init__(self, in_dim, h_dim=0, dropout=0.1):

        super(PositionWiseFeedForwardLayer, self).__init__()
        # hyper-params
        hidden_dim = h_dim if (h_dim > 0) else (4 * in_dim)
        # layers
        self.w1 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.w2 = nn.Linear(hidden_dim, in_dim, bias=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.activation = self.GeLU()
        self.init_weight()

    def init_weight(self):
        torch.nn.init.xavier_uniform_(self.w1.weight.data, gain=1.0)
        torch.nn.init.xavier_uniform_(self.w2.weight.data, gain=1.0)

    def forward(self, x):
        out = self.activation(self.w1(x))
        if self.dropout is not None:
            out = self.dropout(out)
        return self.w2(out)

    class GeLU(nn.Module):
        def forward(self, x):
            return (
                0.5
                * x
                * (
                    1
                    + torch.tanh(
                        math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
                    )
                )
            )


class AddAndNorm(nn.Module):
    def __init__(self, in_dim, dropout=0.1):
        super(AddAndNorm, self).__init__()
        self.norm = nn.LayerNorm(in_dim, eps=1e-06, elementwise_affine=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(self, x, sublayer):
        out = sublayer(self.norm(x))
        if self.dropout is not None:
            out = self.dropout(out)
        return x + out


class TGSAN(BaseModel):
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
        super(TGSAN, self).__init__()
        self.num_labels = num_labels
        # hyper-params
        d_model = 2 * model_config["rnn_hidden_dim"]

        if pretrained_emb is not None:
            _, emb_dim = pretrained_emb.shape
            self.embed = nn.Embedding.from_pretrained(
                torch.tensor(pretrained_emb),
                freeze=(not model_config["embedding_trainable"]),
            )
        else:
            emb_dim = model_config["emb_dim"]
            self.embed = nn.Embedding(num_embeddings=num_emb, embedding_dim=emb_dim)

        self.emb_dropout = (
            nn.Dropout(model_config["embedding_dropout"])
            if model_config["embedding_dropout"] > 0.0
            else None
        )  ## in case of overfitting, 50% neurals will be droped randomly

        # Bi-LSTM encoder
        self.bilstm = nn.LSTM(
            emb_dim,
            model_config["rnn_hidden_dim"],
            1,  # 1 layer
            bidirectional=True,
            batch_first=True,
            dropout=0.0,
        )

        # Target Structured-Self-Attention(r)
        self.tgt_self_san_r = StructuredSelfAttention(
            d_model,
            model_config["tgt_san_dim"],
            r=model_config["r"],
            dropout=model_config["san_dropout"],
            scaled_att=False,
            penal_coeff=model_config["san_penal_coeff"],
        )

        # Context Structured-Attention(r)
        self.ctx_tgt_san_r = StructuredAttention(
            d_model,
            d_model,
            model_config["r"],
            dropout=model_config["san_dropout"],
            scaled_att=False,
            penal_coeff=model_config["san_penal_coeff"],
        )
        # FFN + ADD & LN
        self.ffn = PositionWiseFeedForwardLayer(
            d_model, h_dim=model_config["ffn_dim"], dropout=model_config["ffn_dropout"]
        )
        self.ffn_add_norm = AddAndNorm(d_model, dropout=0.1)
        # Target Structured-Self-Attention(1)
        self.r_mask = nn.Parameter(
            torch.ones(1, model_config["r"], dtype=torch.long), requires_grad=False
        )
        self.tgt_self_san_1 = StructuredSelfAttention(
            d_model,
            model_config["tgt_san_dim"],
            r=1,
            dropout=model_config["san_dropout"],
            scaled_att=False,
            penal_coeff=0.0,
        )
        # Bilinear Attention(1) + Fusion
        self.ctx_tgt_an = BilinearAttention(
            d_model, d_model, scaled_att=True, dropout=model_config["att_dropout"]
        )
        self.ctx_tgt_fuse = nn.Linear(2 * d_model, d_model, bias=True)
        self.ctx_tgt_fuse_active = nn.PReLU()
        # Dense Layer
        self.fc = nn.Linear(d_model, num_labels)
        self.fc_active = nn.PReLU()

        # initialize weights
        self.init_weight()
        self.device = device
        self.to(device)

    def init_weight(self):
        for name, param in self.bilstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param, gain=1.0)
            elif "weight_hh" in name:
                nn.init.xavier_uniform_(param, gain=1.0)

        for name, param in self.fc.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.xavier_uniform_(param, gain=1.0)

        for name, param in self.ctx_tgt_fuse.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.xavier_uniform_(param, gain=1.0)

    def forward(self, raw_text, attention_mask, target_mask, label=None, soft_label=None):
        """
        text: list of token id
        tgt_idx: mask for the target terms
        length_idx: mask for padded sentence
        """
        x = self.embed(raw_text).to(torch.float32)  # [B, L, E]
        if self.emb_dropout is not None:
            x = self.emb_dropout(x)
        len_max = x.size(1)  # embedding dimension
        lens = attention_mask.sum(
            -1
        )  # you can sum across the last array-dimension by using -1
        lens_sort, idx_sort = lens.sort(dim=-1, descending=True)
        idx_ori = idx_sort.sort()[1]
        x = x.index_select(0, idx_sort)
        x = nn.utils.rnn.pack_padded_sequence(
            x, lens_sort, batch_first=True, enforce_sorted=True
        )
        self.bilstm.flatten_parameters()
        x, _ = self.bilstm(x)
        x = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=len_max)[
            0
        ]
        x = x.index_select(0, idx_ori)
        indexed = x * target_mask.unsqueeze(-1).float()
        remain_index = attention_mask - target_mask
        remain = x * remain_index.unsqueeze(-1).float()
        tgt_mat, ctx_mat, ctx_mask = (
            indexed,
            remain,
            remain_index,
        )  # tgt_mat: [B, L, H], ctx_mat: [B, L, H], ctx_mask: [B, L]

        # SCU
        tgt_r, _, tgt_penal = self.tgt_self_san_r(tgt_mat, target_mask)  #  [B, r, H]
        ctx_r, _, ctx_penal = self.ctx_tgt_san_r(
            tgt_r, ctx_mat, ctx_mat, ctx_mask
        )  #  [B, r, H]
        ctx_r = self.ffn_add_norm(ctx_r, lambda _x: self.ffn(_x))
        # CFU
        tgt_vec, _, _ = self.tgt_self_san_1(tgt_r, self.r_mask)  # [B, 1, H]
        ctx_r_fused = torch.cat([ctx_r, tgt_vec.expand_as(ctx_r)], dim=-1)
        ctx_r = ctx_r + self.ctx_tgt_fuse_active(self.ctx_tgt_fuse(ctx_r_fused))
        ctx_vec, _ = self.ctx_tgt_an(tgt_vec, ctx_r, ctx_r, self.r_mask)  # [B, 1, H]

        # OUTPUT
        logits = self.fc_active(self.fc(ctx_vec.squeeze(1)))  # [B, Nc]

        penal_term = tgt_penal
        if ctx_penal is not None:
            penal_term = (
                penal_term + ctx_penal if (penal_term is not None) else ctx_penal
            )
        loss = 0


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
        return loss + penal_term, logits
