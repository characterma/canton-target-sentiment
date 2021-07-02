import torch.nn as nn


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, activation="Tanh"):
        super(FCLayer, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        if activation:
            self.activation = eval(f"nn.{activation}")()
        else:
            self.activation = None

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        if self.activation:
            x = self.activation(x)
        return x


