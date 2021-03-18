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

    INPUT_COLS = []

    def __init__(self):
        super(BaseModel, self).__init__()
        self._caches = defaultdict(lambda: None)
        self.best_state_dict = defaultdict(lambda: None)
        self.best_caches = defaultdict(lambda: None)

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
        print("Device:", self.device)
        self.load_state_dict(
            torch.load(
                state_path,
                # map_location=str(self.device)
            )
        )
