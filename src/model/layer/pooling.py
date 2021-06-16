# -*- coding: utf-8 -*-

# __author__ = Jason Zhang
# __version__ = v0.1


import torch
import torch.nn as nn

class AveragePooling(nn.Module):
    """support mask operation in averge pooling
    mask should in shape [B, L, 1]
    """

    def __init__(self, dim=-1, keep_dim=False):
        super(AveragePooling, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim
    
    def forward(self, x, mask=None):
        if mask is None:
            return x.mean(self.dim, keepdim=self.keep_dim)
        
        _x = x.masked_fill(mask==0, float(0))
        _x = _x.sum(self.dim, keepdim=self.keep_dim)
        return _x / mask.float().sum(1)


class MaxPooling(nn.Module):
    """support mask operation in max pooling
    """

    def __init__(self, dim=-1, keep_dim=False):
        super(MaxPooling, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, x, mask=None):
        if mask is None:
            return x.max(self.dim, keepdim=self.keep_dim)[0]
        
        _x = x.masked_fill(mask==0, float('-inf'))
        return _x.max(self.dim, keepdim=self.keep_dim)[0]


class KMaxPooling(nn.Module):
    """support mask operation in k-max pooling
    """

    def __init__(self, k=2, dim=-1, keep_dim=True):
        super(KMaxPooling, self).__init__()
        self.k = k
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, x, mask=None):
        _x = x if mask is None else x.masked_fill(mask==0, float('-inf'))
        index = _x.topk(self.k, dim = self.dim)[1].sort(dim = self.dim)[0]
        o = x.gather(self.dim, index) # same shape as x except dim=k
        if self.keep_dim:
            return o
        else:
            return o.reshape(x.size(0), -1)
