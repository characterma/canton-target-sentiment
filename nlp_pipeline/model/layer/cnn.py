import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from nlp_pipeline.model.layer.pooling import AveragePooling, MaxPooling, KMaxPooling


class LinearLayer(nn.Module):
    def __init__(self, in_dim, h_dim, activation="PReLU", use_bn=True):
        super(LinearLayer, self).__init__()
        # hyper-params
        if isinstance(h_dim, int):
            h_dim = [h_dim]
        layers = OrderedDict()
        layer_num = len(h_dim) - 1
        # layers
        for i in range(len(h_dim)):
            out_dim = h_dim[i]
            layers[f"fc_{i}"] = nn.Linear(in_dim, out_dim)
            if activation is not None and i != layer_num:
                if use_bn:
                    layers[f"bn_{i}"] = nn.BatchNorm1d(out_dim)
                try:
                    layers[f"activate_{i}"] = eval(f"nn.{activation}")()
                except ValueError as e:
                    raise Exception(
                        f"{str(e)}; pytorch not support activation function: {activation}"
                    )
            in_dim = out_dim
        self.linear = nn.Sequential(layers)

    def forward(self, x):
        o = self.linear(x)
        return o


class ConvLayer(nn.Module):
    """NOTE: a combination of batch normalization and dropout in CNN may have harmful results.
    ref. "Understanding the disharmony between dropout and batch normalization by variance shift"
    pool
    """

    def __init__(
        self,
        in_dim,
        kernel_num=128,
        kernel_sizes=[3, 4, 5],
        activation="ReLU",
        use_bn=True,
        pool_method="max",
        keep_seq_length=False,
        **kwargs,
    ):
        """
        pool_method: ['avg', 'max', 'kmax']
        """

        super(ConvLayer, self).__init__()
        # hyper-params
        if type(kernel_sizes) == int:
            kernel_sizes = [kernel_sizes]
        self.kernel_sizes = kernel_sizes
        self.keep_seq_length = keep_seq_length
        if self.keep_seq_length:
            for k in self.kernel_sizes:
                if k % 2 == 0:
                    raise ValueError(
                        f"Expect all odd kenel sizes when keep_seq_length is True, but got {k}"
                    )
        # layers
        self.kernels = nn.ParameterList()
        self.batch_norms = nn.ModuleList() if use_bn else None
        for i in range(len(kernel_sizes)):
            self.kernels.append(
                nn.Parameter(
                    torch.randn(
                        kernel_num,
                        1,
                        kernel_sizes[i],
                        in_dim,
                        requires_grad=True,
                        dtype=torch.float32,
                    )
                )
            )
            if use_bn:
                self.batch_norms.append(nn.BatchNorm2d(kernel_num))
        self.activation = eval(f"nn.{activation}")() if activation else None
        if (pool_method is not None) and (not self.keep_seq_length):
            if "avg" == pool_method:
                self.pool = AveragePooling(dim=-1, keep_dim=False)
            elif "max" == pool_method:
                self.pool = MaxPooling(dim=-1, keep_dim=False)
            elif "kmax" == pool_method:
                k = kwargs.get("k", 2)
                self.pool = KMaxPooling(k=k, dim=-1, keep_dim=False)
            else:
                raise ValueError("only support 'avg', 'max', 'kmax' pooling methods!")
        else:
            self.pool = None

    def forward(self, x, length_index=None):
        # print("x", x.shape)
        len_max = x.size(1)
        if length_index is not None:
            x = x * length_index.unsqueeze(-1)  # [B, L, H]

        x = x.unsqueeze(1)  # [B, 1, L, H]

        o_all = []
        for i, kernel in enumerate(self.kernels):
            pad_num = (kernel.size(2) // 2, 0) if self.keep_seq_length else 0
            o_i = F.conv2d(
                x, kernel, bias=None, padding=pad_num
            )  # [B, Kn, L, 1] if keep_seq_length else [B, Kn, L-ks+1, 1]
            # print("kernel", kernel.shape)
            # print("o_i", o_i.shape)
            if self.batch_norms:
                o_i = self.batch_norms[i](o_i)
            # print("o_i", o_i.shape)
            o_i = o_i.squeeze(3)  # [B, Kn, L] if keep_seq_length else [B, Kn, L-ks+1]
            if self.activation:
                o_i = self.activation(o_i)
            # print("o_i", o_i.shape)
            if self.pool is not None:
                len_i = length_index[:, : (len_max - self.kernel_sizes[i] + 1)]
                o_i = self.pool(o_i, mask=len_i.unsqueeze(-2))  # [B, Kn*k]
            # print("o_i", o_i.shape)
            o_all.append(o_i)  # [#ks, B, Kn, L] if pool is None else [#ks, B, Kn*k]
        o = (
            torch.cat(o_all, 1) if len(o_all) > 1 else o_all[0]
        )  # [B, Kn*#ks, L] if pool is None else [B, k*Kn*#ks]

        # print(o.shape, length_index.shape)
        if self.keep_seq_length:
            o = o.transpose(1, 2).contiguous() * length_index.unsqueeze(
                -1
            )  # [B, L, Kn*#ks]

        # print("o", o.shape)
        return o


class GRULayer(nn.Module):
    def __init__(
        self,
        in_dim,
        h_dim=128,
        layer_num=1,
        bidirectional=False,
        train_init_step=False,
        dropout=0.0,
    ):
        super(GRULayer, self).__init__()
        # hyper-params
        self.direction_num = 2 if bidirectional else 1
        self.train_h0 = train_init_step
        self.layer_num = layer_num
        self.h_dim = h_dim
        # layers
        self.rnn = nn.GRU(
            in_dim,
            self.h_dim,
            self.layer_num,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout,
        )
        # params
        if self.train_h0:
            self.h0 = nn.Parameter(
                torch.empty(self.direction_num * self.layer_num, 1, self.h_dim),
                requires_grad=True,
            )
            self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.h0.data, gain=nn.init.calculate_gain("tanh"))

    def forward(self, x, length_index):
        batch_size = x.size(0)
        len_max = x.size(1)
        # sort x by lens in decrease order and keep index of original
        lens = length_index.sum(-1)
        lens_sort, idx_sort = lens.sort(dim=-1, descending=True)
        idx_ori = idx_sort.sort()[1]
        x = x.index_select(0, idx_sort)  # [B, L, I]
        x = nn.utils.rnn.pack_padded_sequence(
            x, lens_sort, batch_first=True, enforce_sorted=True
        )
        # continuous memory chunk for cudnn usage
        self.rnn.flatten_parameters()
        o, h_n = (
            self.rnn(x, self.h0.repeat(1, batch_size, 1))
            if self.train_h0
            else self.rnn(x)
        )  # [L, B, Dn*H], [Ln*Dn, B, H]
        # pad packed sequence
        o = nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length=len_max)[
            0
        ]  # [B, L, Dn*H]
        # concat all directions' hidden state of last step in last layer
        h_n = (
            h_n.transpose(0, 1)
            .contiguous()
            .view(batch_size, self.layer_num, self.direction_num, self.h_dim)
        )  # [B, Ln, Dn, H]
        h_n = h_n[:, -1, :, :].contiguous().view(batch_size, -1)  # [B, Dn*H]
        # re-sort by lens
        o = o.index_select(0, idx_ori)
        h_n = h_n.index_select(0, idx_ori)

        return o, h_n


class LSTMLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        h_dim=128,
        layer_num=1,
        bidirectional=False,
        train_init_step=False,
        dropout=0.0,
    ):
        super(LSTMLayer, self).__init__()
        # hyper-params
        self.direction_num = 2 if bidirectional else 1
        self.train_h0 = train_init_step
        self.layer_num = layer_num
        self.h_dim = h_dim
        # layers
        if layer_num > 1:
            self.rnn = nn.LSTM(
                in_dim,
                self.h_dim,
                self.layer_num,
                bidirectional=bidirectional,
                batch_first=True,
                dropout=dropout,
            )
        else:
            self.rnn = nn.LSTM(
                in_dim,
                self.h_dim,
                self.layer_num,
                bidirectional=bidirectional,
                batch_first=True,
                dropout=0,
            )

        # params
        if self.train_h0:
            self.h0 = nn.Parameter(
                torch.empty(self.direction_num * self.layer_num, 1, self.h_dim),
                requires_grad=True,
            )
            self.c0 = nn.Parameter(
                torch.empty(self.direction_num * self.layer_num, 1, self.h_dim),
                requires_grad=True,
            )
            self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.h0.data, gain=nn.init.calculate_gain("tanh"))
        nn.init.xavier_normal_(self.c0.data, gain=nn.init.calculate_gain("tanh"))

    def forward(self, x, length_index):
        batch_size, len_max, _ = x.size()

        # sort x by lens in decrease order and keep index of original
        lens = length_index.sum(-1)
        lens_sort, idx_sort = lens.sort(dim=-1, descending=True)
        idx_ori = idx_sort.sort()[1]
        x = x.index_select(0, idx_sort)  # [B, L, I]
        x = nn.utils.rnn.pack_padded_sequence(
            x, lens_sort, batch_first=True, enforce_sorted=True
        )
        # continuous memory chunk for cudnn usage
        self.rnn.flatten_parameters()
        o, h_c_n = (
            self.rnn(
                x, (self.h0.repeat(1, batch_size, 1), self.c0.repeat(1, batch_size, 1))
            )
            if self.train_h0
            else self.rnn(x)
        )  # [L, B, Dn*H], tupel([Ln*Dn, B, H], [Ln*Dn, B, H])
        # pad packed sequence
        o = nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length=len_max)[
            0
        ]  # [B, L, Dn*H]
        # concat all directions' hidden state of last step in last layer
        h_n = (
            h_c_n[0]
            .view(self.layer_num, self.direction_num, batch_size, self.h_dim)[
                -1, :, :, :
            ]
            .transpose(0, 1)
            .contiguous()
            .view(batch_size, -1)
        )  # [B, Dn*H]
        # re-sort by lens
        o = o.index_select(
            0, idx_ori
        )  # [[[1f,1b][2f,2b]...[max_Lenf,max_Lenb]]] used in sequences tagging
        h_n = h_n.index_select(0, idx_ori)  # [[lenF,1B]] used in classification

        return o, h_n
