# coding=utf-8
import argparse
import json
import logging
import os
import pandas as pd
import shutil
from pathlib import Path
from trainer import Trainer, evaluate
from utils import set_seed, set_log_path, load_config, save_config, load_yaml, log_args
from dataset import get_dataset
from tokenizer import get_tokenizer
from model import get_model


logger = logging.getLogger(__name__)


def run(args):
    tokenizer = get_tokenizer(args=args)
    model = get_model(args=args)
    datasets = []
    if not args.test_only:
        train_dataset = get_dataset(dataset="train", tokenizer=tokenizer,args=args)
        dev_dataset = get_dataset(dataset="dev", tokenizer=tokenizer, args=args)
        trainer = Trainer(model=model, train_dataset=train_dataset, dev_dataset=dev_dataset, args=args)
        trainer.train()
    else:
        train_dataset = None
        dev_dataset = None

    test_dataset = get_dataset(dataset="test", tokenizer=tokenizer, args=args)

    if not args.test_only:
        train_metrics = evaluate(model=model, eval_dataset=train_dataset, args=args,)
        dev_metrics = evaluate(model=model, eval_dataset=dev_dataset, args=args)
    else:
        train_metrics = None
        dev_metrics = None

    test_metrics = evaluate(model=model, eval_dataset=test_dataset, args=args)
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
