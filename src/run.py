# coding=utf-8
import logging
from pathlib import Path
import os, shutil, sys
import uuid
import numpy as np
import pickle

import argparse
import pandas as pd
from dataset import TargetDependentDataset
from trainer import Trainer, evaluate
from transformers_utils import PretrainedBert
from utils import (
    init_logger,
    set_seed,
    load_yaml,
    save_yaml,
    get_label_map,
    Timer,
    MODEL_EMB_TYPE,
)
from tokenizer import get_tokenizer
from model import *
import json
from gensim.models import KeyedVectors

logger = logging.getLogger(__name__)


def set_log_path():
    logging.basicConfig(
        handlers=[logging.FileHandler(Path("../log/log"), "w+", "utf-8"), logging.StreamHandler()], 
        format="%(message)s", 
        level=logging.INFO
    )


def load_config(args):
    config_dir = Path(args.config_dir)
    run_config = load_yaml(config_dir / "run.yaml")
    args.data_config = run_config['data']
    args.eval_config = run_config['eval']

    if args.test_only:
        in_dir = Path("../output") / args.data_config['model_dir'] / 'train'
        if in_dir.exists() and in_dir.isdir():
            run_config_tr = load_yaml(in_dir / "run.yaml")
            args.train_config = run_config_tr['train']
            args.prepro_config = run_config_tr['text_prepro']
            model_config = load_yaml(in_dir / "model.yaml")
        else:
            print("Experiment not found.")
            raise
    else:
        args.train_config = run_config['train']
        args.prepro_config = run_config['text_prepro']
        model_config = load_yaml(config_dir / "model.yaml")
    
    model_class = args.train_config['model_class']
    args.model_config = model_config[model_class]
    return args


def init_model(args):
    model_class = args.train_config['model_class']
    MODEL = getattr(sys.modules[__name__], model_class)
    model = MODEL(args=args)
    if args.test_only:
        in_dir = Path("../output") / args.data_config['model_dir'] / 'train'
        model.load_state(in_dir / args.eval_config["state_file"])
    return model


def run_bert(args):
    """

    """
    model = init_model(args)
    tokenizer = get_tokenizer(args=args)

    # if non_bert:
    #     word_to_idx = TargetDependentDataset.build_vocab(
    #         dataset="train",
    #         tokenizer=tokenizer,
    #         args=args
    #     ) # keep top 95% frequent tokens
    # else:
    #     word_to_idx=None

    if not args.test_only:

        train_dataset = TargetDependentDataset(
            dataset="train",
            tokenizer=tokenizer,
            word_to_idx=None, 
            args=args
        )

        dev_dataset = TargetDependentDataset(
            dataset="dev",
            tokenizer=tokenizer,
            word_to_idx=None, 
            args=args
        )

        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            args=args,
        )

        trainer.train()

    test_dataset = TargetDependentDataset(
        dataset="test",
        tokenizer=tokenizer,
        args=args
    )

    metrics = evaluate(
        model=model,
        eval_dataset=test_dataset,
        args=args,
    )

    eval_out_dir = Path("../output") / args.data_config['model_dir'] / 'eval'
    json.dump(metrics, eval_out_dir / 'metrics.json')


if __name__ == "__main__":
    set_log_path()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="../config/")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    args = load_config(args)
    set_seed(args.train_config["seed"])
    # args.preprocess_mode = "unit_content" 
    run_bert(args=args)
