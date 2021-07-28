import torch.nn as nn
from collections import OrderedDict


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, activation=None):
        super(FCLayer, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = None
        if activation:
            self.activation = eval(f"nn.{activation}")()

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        if self.activation:
            x = self.activation(x)
        return x


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
