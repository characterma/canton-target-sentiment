import torch
import numpy as np
import importlib
from inspect import signature


def get_model_inputs(args):
    model_class = args.train_config["model_class"]
    Model = getattr(importlib.import_module(f"model.{args.task}"), model_class)
    sig = signature(Model.forward)
    return list(sig.parameters.keys())


def pad_array(array, max_length, value=0):
    d = max_length - len(array)
    if d >= 0:
        array = np.concatenate((array, [value] * d), axis=None)
        return array
    else:
        raise Exception("Array length should not exceed max_length.")


def pad_tensor(tensor, pad, dim):
    # tensor: tensor or np.array, pad: int
    pad_size = list(tensor.shape)
    pad_size[dim] = pad - tensor.size(dim)
    return torch.cat([tensor, torch.zeros(*pad_size).long()], dim=dim)


class PadCollate:
    def __init__(self, pad_cols, input_cols, dim=0):
        self.dim = dim
        self.pad_cols = pad_cols
        self.input_cols = input_cols

    def pad_collate(self, batch):
        outputs = dict()
        for col in self.input_cols:
            if col in self.pad_cols:
                max_len = max(map(lambda x: x[col].shape[self.dim], batch))
                x_col = list(
                    map(lambda x: pad_tensor(x[col], pad=max_len, dim=self.dim), batch)
                )
                x_col = torch.stack(x_col, dim=0)
            else:
                x_col = torch.stack(
                    list(map(lambda x: torch.tensor(x[col]), batch)), dim=0
                )
            outputs[col] = x_col
        return outputs

    def __call__(self, batch):
        return self.pad_collate(batch)
