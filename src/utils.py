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


def get_label_to_id(args):
    task = args.run_config['train']['task']
    label_to_id_path = args.model_dir / "label_to_id.json"
    if os.path.exists(label_to_id_path):
        label_to_id = json.load(open(label_to_id_path, 'r'))
        return label_to_id
    if task=='target_sentiment':
        labels = args.run_config['data']['labels']
        if labels=="2_ways":
            label_to_id = {"neutral": 0, "non_neutral": 1}
        elif labels=="3_ways":
            label_to_id = {"neutral": 0, "negative": 1, "positive": 2}
        else:
            raise ValueError("Label type not supported.")

    elif task=='chinese_word_segmentation':
        # TODO: use get_token_level_tags to extract tags in train data; save label_to_id
        label_to_id = {'X': 0}
        tags = []
        # Scan all data and get tags
        files = []
        for dataset in ['train', 'dev', 'test']:
            files.append(args.data_config[dataset])
        files = list(set(files))
        
        for filename in files:
            data_path = args.data_dir / filename
            raw_data = json.load(open(data_path, 'r'))
            for x in raw_data:
                tags.extend(list(set(x['postags'])))

        tags = list(set(tags))
        for t in tags:
            for suffix in ['B', 'I']:
                tag = f"{suffix}-{t}"
                label_to_id[tag] = len(label_to_id)

    label_to_id_inv = dict(zip(label_to_id.values(), label_to_id.keys()))
    print(label_to_id)
    return label_to_id, label_to_id_inv



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
    args.label_to_id, args.label_to_id_inv = get_label_to_id(args)
    return args


def save_config(args):
    if not os.path.exists(args.model_dir / "run.yaml"):
        shutil.copy(args.config_dir / "run.yaml", args.model_dir / "run.yaml")
    if not os.path.exists(args.model_dir / "model.yaml"):
        shutil.copy(args.config_dir / "model.yaml", args.model_dir / "model.yaml")




