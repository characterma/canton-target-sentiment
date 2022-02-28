import torch
import numpy as np
import importlib
from inspect import signature


def get_model_inputs(args):
    model_class = args.train_config["model_class"]
    Model = getattr(importlib.import_module(f"nlp_pipeline.model.{args.task}"), model_class)
    sig = signature(Model.forward)
    return list(sig.parameters.keys())


def pad_array(array, max_length, value=0):
    d = max_length - len(array)
    if d >= 0:
        array = np.concatenate((array, [value] * d), axis=None)
        return array
    else:
        raise Exception("Array length should not exceed max_length.")


def pad_tensor(tensor, pad, pad_dim):
    # tensor: tensor or np.array, pad: int
    pad_size = list(tensor.shape)
    pad_size[pad_dim] = pad - tensor.size(pad_dim)
    return torch.cat([tensor, torch.zeros(*pad_size).long()], dim=pad_dim)


class PadCollate:
    def __init__(self, pad_cols, pad_dim=0, max_length=None):
        self.pad_dim = pad_dim
        self.pad_cols = pad_cols
        self.max_length = max_length

    def pad_collate(self, batch):
        outputs = dict()
        input_cols = list(batch[0].keys())
        for col in input_cols:
            if col in self.pad_cols:
                if self.max_length is not None:
                    target_len = max(map(lambda x: x[col].shape[self.pad_dim], batch))
                else:
                    target_len = self.max_length
                
                x_col = list(
                    map(lambda x: pad_tensor(x[col], pad=target_len, pad_dim=self.pad_dim), batch)
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
