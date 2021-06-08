# coding=utf-8
import argparse
import json
import logging
import os
import pandas as pd
import shutil
from pathlib import Path
from dataset import TargetDependentDataset, build_vocab_from_dataset, build_vocab_from_pretrained, load_vocab
from trainer import Trainer, evaluate
from utils import set_seed, set_log_path, load_config, save_config, load_yaml, log_args
from tokenizer import get_tokenizer
from model import get_model, get_model_type


logger = logging.getLogger(__name__)


def init_model(args):
    model_path = args.model_dir / args.eval_config["model_file"]
    if not os.path.exists(model_path):
        model_path = None 
    model = get_model(args, model_path=model_path)
    return model


def init_tokenizer(args):
    tokenizer = get_tokenizer(args=args)
    model_type = get_model_type(args=args)
    if model_type=="non_bert":
        vocab_path = args.model_dir / 'word_to_idx.json'
        if not os.path.exists(vocab_path):
            if args.model_config.get("pretrained_emb", None) is not None:
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

        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            args=args,
        )

        trainer.train()
    else:
        train_dataset = None
        dev_dataset = None

    test_dataset = TargetDependentDataset(
        dataset="test",
        tokenizer=tokenizer,
        args=args
    )

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
    else:
        train_metrics = None
        dev_metrics = None

    test_metrics = evaluate(
        model=model,
        eval_dataset=test_dataset,
        args=args,
    )

    combine_and_save_metrics(metrics=[train_metrics, dev_metrics, test_metrics], args=args)
    combine_and_save_statistics(datasets=[train_dataset, dev_dataset, test_dataset], args=args)
    save_config(args)


def combine_and_save_metrics(metrics, args):
    metrics = [m for m in metrics if m is not None]
    metrics_df = pd.DataFrame(data=metrics)
    filename = 'result_test_only.csv' if args.test_only else 'result.csv'
    metrics_df.to_csv(args.result_dir / filename, index=False)


def combine_and_save_statistics(datasets, args):
    datasets = [ds for ds in datasets if ds is not None]
    diagnosis_df = pd.concat([ds.diagnosis_df for ds in datasets])
    filename = 'diagnosis_test_only.xlsx' if args.test_only else 'diagnosis.xlsx'
    diagnosis_df.to_excel(args.result_dir / filename, index=False)
    statistics_df = pd.DataFrame(data=[ds.get_data_analysis() for ds in datasets])
    filename = 'statistics_test_only.csv' if args.test_only else 'statistics.csv'
    statistics_df.to_csv(args.result_dir / filename, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="../config/")
    parser.add_argument("--test_only", action="store_true")
    args = parser.parse_args()

    if args.test_only:
        run_config = load_yaml(Path(args.config_dir) / "run.yaml")
        args.config_dir = Path(run_config['data']['output_dir'])

    args = load_config(args)
    set_log_path(args.output_dir)
    log_args(logger, args)
    set_seed(args.train_config["seed"])
    run(args=args)
