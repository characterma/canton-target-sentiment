import torch
import torch.nn as nn
import copy
import os
import numpy as np
import logging
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


class BaseModel(nn.Module):
    """
    Common methods:
    1. save / load state
    2. cache / save variables
    3. basic loggging
    """
    def __init__(self):
        super(BaseModel, self).__init__()
        self._caches = defaultdict(lambda: None)
        self.best_state_dict = defaultdict(lambda: None)
        self.best_caches = defaultdict(lambda: None)

    def mark_as_best(self, name, cache_name=""):
        """
        Store the states and caches of the best model.
        """
        self.best_state_dict[name] = copy.deepcopy(self.state_dict())
        if cache_name:
            self.best_caches[name] = {
                "cache_name": cache_name, 
                "cache_data": copy.deepcopy(self._caches[cache_name])
            }
            

    def load_best_state(self, name):
        if self.best_state_dict[name] is not None:
            self.load_state_dict(self.best_state_dict[name])

    def save_best_state(self, name, output_dir, filename="model_state.pt"):
        logger.info("***** Saving best model states *****")
        logger.info("  Directory = %s", str(output_dir))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_dir = Path(output_dir)
        if filename:
            state_path = output_dir / filename
        else:
            state_path = output_dir / filename
        if self.best_state_dict is not None:
            torch.save(self.best_state_dict[name], state_path)

    def load_state(self, state_path):
        if "*." in state_path.name:
            file_ext = "." + state_path.name.split(".")[-1]
            for f in list(state_path.parent.glob("*"))[
                -1::-1
            ]:  # use the last saved model
                if f.name.endswith(file_ext):
                    state_path = f
                    break
        logger.info("***** Loading model state *****")
        logger.info("  Path = %s", str(state_path))
        assert state_path.is_file()
        self.load_state_dict(torch.load(state_path))

    def cache(self, cache_name, tensor_name, tensor):
        if cache_name not in self._caches:
            self._caches[cache_name] = dict()
        if tensor_name not in self._caches[cache_name]:
            # Convert tensor into np.array: https://blog.csdn.net/moshiyaofei/article/details/90519430
            self._caches[cache_name][tensor_name] = np.atleast_1d(tensor.cpu().numpy())
        else:
            self._caches[cache_name][tensor_name] = np.concatenate(
                (self._caches[cache_name][tensor_name], np.atleast_1d(tensor.cpu().numpy()))
            )

    def clear_caches(self):
         self._caches = dict()

    def save_caches(self, output_dir):
        """
        Save each variable as individual npy file. Reason see https://stackoverflow.com/questions/9619199/best-way-to-preserve-numpy-arrays-on-disk
        """
        logger.info("***** Saving caches *****")
        logger.info("  Directory = %s", str(output_dir))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for dataset_name in self.best_caches.keys():
            cache_name = self.best_caches[dataset_name]["cache_name"]
            cache_data = self.best_caches[dataset_name]["cache_data"]
            for tensor_name, matrix in cache_data.items():
                # matrix: row=sample, column=feature dim
                np.save(
                    Path(output_dir) / f"{cache_name}_{tensor_name}.npy", 
                    matrix
                )





