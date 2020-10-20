import logging
import os
from pathlib import Path
import random
import json
from argparse import Namespace

import numpy as np
import pandas as pd
import torch

from sklearn import metrics
from ruamel.yaml import YAML


def get_label(data_config):
    label_path = Path(os.environ['CANTON_SA_DIR']) / "data" / data_config["data_path"]["subfolder"] / data_config["data_path"]["label"]
    return [
        label.strip()
        for label in open(
            label_path, "r", encoding="utf-8"
        )
    ]


def write_eval_details(data_config, eval_details_path, details, preds, keys):
    """
    For official evaluation script
    :param output_file: prediction_file_path (e.g. eval/proposed_answers.txt)
    :param preds: [0,1,0,2,18,...]
    """
    relation_labels = get_label(data_config)
    preds = pd.DataFrame(data={"key": keys, 
                               "pred": [relation_labels[p] for p in preds]
                               })
    details = details.set_index("key")
    preds = preds.set_index("key")
    details = details.join(preds, how='inner')
    details.to_csv(eval_details_path, index=True, encoding='utf-8-sig')
    
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


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels, average="macro"):
    acc = simple_accuracy(preds, labels)
    return {
        "acc": acc,
        "macro_f1": metrics.f1_score(labels, preds, labels=[0, 1, 2], average='macro'),
        "micro_f1": metrics.f1_score(labels, preds, labels=[0, 1, 2], average='micro'),
    }


def save_yaml(data, file_path):
    yaml = YAML()
    yaml.dump(data, file_path)


def load_yaml(file_path):
    yaml = YAML()
    data = yaml.load(open(file_path, 'r'))
    return data