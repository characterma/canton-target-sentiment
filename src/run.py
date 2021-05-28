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


def save_config(args):
    if not os.path.exists(args.model_dir / "run.yaml"):
        shutil.copy(args.config_dir / "run.yaml", args.model_dir / "run.yaml")
    if not os.path.exists(args.model_dir / "model.yaml"):
        shutil.copy(args.config_dir / "model.yaml", args.model_dir / "model.yaml")


def init_model(args):
    state_path = args.model_dir / args.eval_config["state_file"]
    if not os.path.exists(state_path):
        state_path = None 
    model = get_model(args, state_path=state_path)
    return model


def init_tokenizer(args):
    tokenizer = get_tokenizer(args=args)
    model_type = get_model_type(args=args)
    if model_type=="non_bert":
        vocab_path = args.model_dir / 'word_to_idx.json'
        if not os.path.exists(vocab_path):
            if args.word_embedding_path is not None:
                build_vocab_from_pretrained(tokenizer=tokenizer, args=args)
            else:
                build_vocab_from_dataset(dataset="train", tokenizer=tokenizer, args=args)
        else:
            load_vocab(tokenizer=tokenizer, vocab_path=vocab_path, args=args)
    return tokenizer


def run(args):
    tokenizer = init_tokenizer(args=args)
    model = init_model(args=args)
    datasets = []

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

        datasets.extend([train_dataset, dev_dataset])

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
    datasets.append(test_dataset)

    metrics = []

    if not args.test_only:
        train_metrics = evaluate(
            model=model,
            eval_dataset=train_dataset,
            args=args,
        )
        dev_metrics = evaluate(
            model=model,
            eval_dataset=dev_dataset,
            args=args,
        )
        metrics.extend([train_metrics, dev_metrics])

    test_metrics = evaluate(
        model=model,
        eval_dataset=test_dataset,
        args=args,
    )
    metrics.append(test_metrics)

    combine_and_save_metrics(metrics=metrics, args=args)
    combine_and_save_diagnosis(datasets=datasets, args=args)
    combine_and_save_statistics(datasets=datasets, args=args)
    save_config(args)


def combine_and_save_metrics(metrics, args):
    metrics_df = pd.DataFrame(data=metrics)
    metrics_df.to_csv(args.model_dir / 'result.csv', index=False)


def combine_and_save_diagnosis(datasets, args):
    diagnosis_df = pd.concat([ds.diagnosis_df for ds in datasets])
    diagnosis_df.to_excel(args.model_dir / 'diagnosis.xlsx', index=False)


def combine_and_save_statistics(datasets, args):
    statistics_df = pd.DataFrame(data=[ds.get_data_analysis() for ds in datasets])
    statistics_df.to_csv(args.model_dir / 'statistics.csv', index=False)


if __name__ == "__main__":
    set_log_path()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="../config/")
    parser.add_argument("--test_only", action="store_true")
    args = parser.parse_args()

    if args.test_only:
        run_config = load_yaml(Path(args.config_dir) / "run.yaml")
        args.config_dir = Path(run_config['data']['model_dir'])

    args = load_config(args)
    set_seed(args.train_config["seed"])
    run(args=args)
