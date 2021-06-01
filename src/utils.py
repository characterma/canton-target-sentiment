# coding=utf-8
import logging
import numpy as np
import os
import pandas as pd
import random
import shutil
import torch
import yaml
from argparse import Namespace
from pathlib import Path


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(file_path):
    data = yaml.load(open(file_path, "r"), Loader=yaml.FullLoader)
    return data


def set_log_path():
    logging.basicConfig(
        handlers=[logging.FileHandler(Path("../log/log"), "w+", "utf-8"), logging.StreamHandler()], 
        format="%(message)s", 
        level=logging.INFO
    )


def get_label_to_id(labels):
    if labels=="2_ways":
        label_to_id = {"neutral": 0, "non_neutral": 1}
    elif labels=="3_ways":
        label_to_id = {"neutral": 0, "negative": 1, "positive": 2}
    else:
        raise ValueError("Label type not supported.")
    label_to_id_inv = dict(zip(label_to_id.values(), label_to_id.keys()))
    return label_to_id, label_to_id_inv


def load_config(args):
    config_dir = Path(args.config_dir)
    run_config = load_yaml(config_dir / "run.yaml")
    args.data_config = run_config['data']
    args.label_to_id, args.label_to_id_inv = get_label_to_id(run_config['data']['labels'])
    args.eval_config = run_config['eval']
    args.device = run_config['device']
    args.train_config = run_config['train']
    args.prepro_config = run_config['text_prepro']
    model_config = load_yaml(config_dir / "model.yaml")
    model_class = args.train_config['model_class']
    args.model_config = model_config[model_class]
    args.model_config.update(run_config['model_params'])
    args.config_dir = config_dir
    model_dir = Path(args.data_config['model_dir'])
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    args.model_dir = model_dir
    return args




