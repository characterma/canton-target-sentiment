# coding=utf-8
import logging
import os
from pathlib import Path
import random
import json
from argparse import Namespace
from ruamel.yaml.comments import CommentedMap
import numpy as np
import pandas as pd
import torch
import time

from ruamel.yaml import YAML
from sklearn.model_selection import ParameterGrid


SENTI_ID_MAP = {
    "neutral": 0,
    "negative": 1,
    "positive": 2,
}

SENTI_ID_MAP_INV = {}
for k, v in SENTI_ID_MAP.items():
    SENTI_ID_MAP_INV[v] = k

MODEL_EMB_TYPE = {
    "TGSAN": "WORD",
    "TDLSTM": "WORD",
    "ATAE_LSTM": "WORD",
    "IAN": "WORD",
    "MEMNET": "WORD",
    "RAM": "WORD",
    "TNET_LF": "WORD",
    "TDBERT": "BERT",
}


class SPEC_TOKEN:
    TARGET = "[TGT]"


def get_label_map(label_map_path):
    with open(label_map_path, encoding="utf-8") as json_file:
        data = json.load(json_file)
        return data


def join_eval_details(data_config, details, preds, keys):
    """
    For official evaluation script
    :param output_file: prediction_file_path (e.g. eval/proposed_answers.txt)
    :param preds: [0,1,0,2,18,...]
    """
    # relation_labels = get_label(data_config)
    preds = pd.DataFrame(
        data={"key": keys, "pred": [SENTI_ID_MAP_INV[p] for p in preds]}
    )
    details = details.set_index("key")
    preds = preds.set_index("key")
    details = details.join(preds, how="inner")
    return details
    #


def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        

def save_yaml(data, file_path):
    yaml = YAML()
    yaml.dump(data, file_path)


def load_yaml(file_path, overwriting_config=None):
    yaml = YAML()
    data = yaml.load(open(file_path, "r"))
    if isinstance(overwriting_config, CommentedMap) or isinstance(
        overwriting_config, dict
    ):
        data = apply_overwriting_config(overwriting_config, data)
    return data


def generate_grid_search_params(grid_config):
    param_grid = dict()
    body_config = grid_config["body"]
    optim_config = grid_config["optim"]
    for k, v in body_config.items():
        param_grid[f"body:::{k}"] = v
    for k, v in optim_config.items():
        param_grid[f"optim:::{k}"] = v
    param_comb = list(ParameterGrid(param_grid))
    return param_comb


def apply_overwriting_config(overwriting_config, config):
    for k, v in overwriting_config.items():
        if k in config:
            if isinstance(v, CommentedMap) or isinstance(v, dict):
                config[k] = apply_overwriting_config(v, config[k])
            else:
                config[k] = v
        else:
            config[k] = v
    return config


def apply_grid_search_params(params, body_config, optim_config):
    """
    overwrite a dict
    """
    for k, v in params.items():
        p_type, p_name = k.split(":::")
        if p_type == "body":
            body_config[p_name] = v
        elif p_type == "optim":
            optim_config[p_name] = v
        else:
            assert False
    return body_config, optim_config


def parse_api_req(req_dict):
    left_sep = "## Headline ##\n"
    right_sep = "\n## Content ##\n"

    output_dict = {}
    if req_dict["target_in_hl"] == 0: # target in content
        hl_with_sep = left_sep + req_dict["headline"] + right_sep
        output_dict["content"] = hl_with_sep + req_dict["content"]
        output_dict["start_ind"] = req_dict["start_ind"] + len(hl_with_sep)
        output_dict["end_ind"] = req_dict["end_ind"] + len(hl_with_sep)
    else:
        hl_with_sep = left_sep + req_dict["headline"] + right_sep
        output_dict["content"] = hl_with_sep + req_dict["content"]
        output_dict["start_ind"] = req_dict["start_ind"] + len(left_sep)
        output_dict["end_ind"] = req_dict["end_ind"] + len(left_sep)       

    return output_dict 


class Timer:
    def __init__(self, output_dir):
        self.preprocessing_start_time = None
        self.inference_start_time = None
        self.durations = {
            "preprocessing": 0, 
            "inference": 0,  
        }
        self.output_dir = output_dir

    def on_preprocessing_start(self):
        self.preprocessing_start_time = time.time()

    def on_preprocessing_end(self):
        self.durations["preprocessing"] += time.time() - self.preprocessing_start_time

    def on_inference_start(self):
        self.inference_start_time = time.time()

    def on_inference_end(self):
        self.durations["inference"] += time.time() - self.inference_start_time

    # @property
    # def durations(self):
    #     return self.durations

    def save_timer(self):
        with open(self.output_dir / "timer.json", 'w') as fp:
            json.dump(self.durations, fp)



