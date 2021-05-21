# coding=utf-8
import logging
from pathlib import Path
import os, shutil, sys
import uuid
import numpy as np
import pickle

import argparse
import pandas as pd
from dataset import TargetDependentDataset, build_vocab_from_dataset, build_vocab_from_pretrained, load_vocab
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
from model import get_model, get_model_type
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
        config_dir = Path("../output") / args.data_config['model_dir']
        run_config = load_yaml(config_dir / "run.yaml")

    args.train_config = run_config['train']
    args.prepro_config = run_config['text_prepro']
    model_config = load_yaml(config_dir / "model.yaml")
    model_class = args.train_config['model_class']
    args.model_config = model_config[model_class]

    model_dir = Path("../output") / args.data_config['model_dir']
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    args.model_dir = model_dir

    return args


def save_config(args):
    if not os.path.exists(args.model_dir / "run.yaml"):
        shutil.copy(args.config_dir / "run.yaml", args.model_dir / "run.yaml")
    if not os.path.exists(args.model_dir / "model.yaml"):
        shutil.copy(args.config_dir / "model.yaml", args.model_dir / "model.yaml")


def init_model(args):
    if args.test_only:
        state_path = Path("../output") / args.data_config['model_dir'] / args.eval_config["state_file"]
    else:
        state_path = None 
    model = get_model(args, state_path=state_path)
    return model


def init_tokenizer(args):
    tokenizer = get_tokenizer(args=args)
    model_type = get_model_type(args=args)
    if model_type=="non_bert":
        if not args.test_only:
            if args.word_embedding_path is not None:
                build_vocab_from_pretrained(tokenizer=tokenizer, args=args)
            else:
                build_vocab_from_dataset(dataset="train", tokenizer=tokenizer, args=args)
        else:
            load_vocab(tokenizer=tokenizer, args=args)
    return tokenizer


def run(args):
    tokenizer = init_tokenizer(args=args)
    model = init_model(args=args)

    if not args.test_only:
        train_dataset = TargetDependentDataset(
            dataset="train",
            tokenizer=tokenizer,
            args=args
        )

        dev_dataset = TargetDependentDataset(
            dataset="dev",
            tokenizer=tokenizer,
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

    dev_metrics = evaluate(
        model=model,
        eval_dataset=dev_dataset,
        args=args,
    )
    test_metrics = evaluate(
        model=model,
        eval_dataset=test_dataset,
        args=args,
    )
    # 

    results = {
        'dev': dev_metrics, 
        'test': test_metrics
    }
    statistics = {
        'train': train_dataset.statistics, 
        'dev': dev_metrics.statistics, 
        'test': test_dataset.statistics
    }
    json.dump(results, open(args.model_dir / 'results.json', 'w'))
    json.dump(statistics, open(args.model_dir / 'data_statistics.json', 'w'))
    save_config(args)


if __name__ == "__main__":
    set_log_path()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="../config/")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    args = load_config(args)
    set_seed(args.train_config["seed"])
    run(args=args)
