# coding=utf-8
import logging
import numpy as np
import os
import json
import pandas as pd
import random
import shutil
import torch
import transformers 
import yaml
from argparse import Namespace
from pathlib import Path


def log_args(logger, args):
    logger.info("***** Args *****")
    for k1, v1 in args.run_config.items():
        logger.info(f"   {k1}: {str(v1)}")


def set_seed(seed):
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(file_path):
    data = yaml.load(open(file_path, "r"), Loader=yaml.FullLoader)
    return data


def set_log_path(log_dir):
    logging.basicConfig(
        handlers=[logging.FileHandler(log_dir / "log", "w+", "utf-8"), logging.StreamHandler()], 
        format="%(message)s", 
        level=logging.INFO
    )


def load_config(args):
    config_dir = Path(args.config_dir)
    run_config = load_yaml(config_dir / "run.yaml")
    args.run_config = run_config
    args.data_config = run_config['data']
    args.eval_config = run_config['eval']
    args.device = run_config['device']
    args.train_config = run_config['train']
    args.prepro_config = run_config['text_prepro']
    model_config = load_yaml(config_dir / "model.yaml")
    model_class = args.train_config['model_class']
    args.model_config = model_config[model_class]
    args.model_config.update(run_config['model_params'])
    args.config_dir = config_dir
    output_dir = Path(args.data_config['output_dir'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    args.output_dir = output_dir
    args.data_dir = Path(args.data_config["data_dir"])
    args.model_dir = output_dir / "model"
    args.result_dir = output_dir / "result"
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    args.pretrained_emb_path = args.model_config.get("pretrained_emb_path", None)
    return args


def save_config(args):
    if not os.path.exists(args.model_dir / "run.yaml"):
        shutil.copy(args.config_dir / "run.yaml", args.model_dir / "run.yaml")
    if not os.path.exists(args.model_dir / "model.yaml"):
        shutil.copy(args.config_dir / "model.yaml", args.model_dir / "model.yaml")




