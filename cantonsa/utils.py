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

from sklearn import metrics
from ruamel.yaml import YAML
from cantonsa.constants import SENTI_ID_MAP_INV
from sklearn.model_selection import ParameterGrid


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


def compute_metrics(preds, labels, docids=None):
    assert len(preds) == len(labels)
    acc_rep = metrics.classification_report(
        labels, preds, labels=["neutral", "negative", "positive"], output_dict=True
    )
    output = {
        "acc": (preds == labels).mean(),
        "macro_f1": metrics.f1_score(
            labels, preds, labels=["neutral", "negative", "positive"], average="macro"
        ),
        "micro_f1": metrics.f1_score(
            labels, preds, labels=["neutral", "negative", "positive"], average="micro"
        ),
        "ars": ars(preds, labels, docids) if docids is not None else None,
    }
    # print(acc_rep)
    for class_name, v1 in acc_rep.items():
        if class_name in ["neutral", "negative", "positive"]:
            for score_name, v2 in v1.items():
                output[f"{class_name}-{score_name}"] = v2
    return output


def ars(preds, labels, docids):
    """
    ARS from https://arxiv.org/abs/2009.07964
    keys: the unique id of each example
    """
    df = pd.DataFrame(data={"pred": preds, "label": labels, "docid": docids})
    df["correct"] = (df["pred"] == df["label"]).astype(int)
    df["count"] = 1
    df = df.groupby(by="docid").agg({"correct": sum, "count": sum})

    df["correct_ars"] = (df["correct"] == df["count"]).astype(int)
    return df["correct_ars"].sum() / df["correct_ars"].shape[0]


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
