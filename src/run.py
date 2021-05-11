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
from trainer import Trainer
from evaluater import Evaluater
from transformers_utils import PretrainedLM
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


def init_bert_model(args):
    pretrained_lm = PretrainedLM(args.model_config[model_class]["pretrained_lm"])
    MODEL = getattr(sys.modules[__name__], model_class)
    model = BERT_MODEL(args=args)
    if state_path is not None:
        model.load_state(state_path)
    return model


def run_bert_unit_content(args):

    args.preprocess_mode = "unit_content"

    if args.train_config["state_file"]:
        train_state_path = (
            args.base_dir / "output" / "train" / args.train_config["state_file"]
        )
    else:
        train_state_path = None
    train_output_dir = args.base_dir / "output" / "train" / args.train_config["output_dir"]

    if not os.path.exists(train_output_dir):
        os.makedirs(train_output_dir)

    eval_input_dir = args.eval_config["input_dir"]
    model_class = args.train_config["model_class"]
    set_seed(args.train_config["seed"])

    dataset_dir = args.base_dir / "data" / "datasets" / args.data_config["dataset"]

    # load pretrained language model
    tokenizer = get_tokenizer(
        source=args.model_config[model_class]["tokenizer_source"],
        name=args.model_config[model_class]["tokenizer_name"],
    )

    model = init_bert_model(args)

    if args.do_train:

        train_dataset = TargetDependentDataset(
            dataset_dir / data_config["train"],
            tokenizer,
            preprocess_config=rgs.model_config[model_class],
            required_features=model.INPUT_COLS, 
        )

        dev_dataset = TargetDependentDataset(
            dataset_dir / data_config["dev"],
            tokenizer,
            preprocess_config=rgs.model_config[model_class],
            required_features=model.INPUT_COLS, 
        )

        trainer = Trainer(
            model=model,
            output_dir=train_output_dir,
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            args=args,
        )

        trainer.train()

    if args.do_test:

        test_dataset = TargetDependentDataset(
            dataset_dir / data_config["test"],
            tokenizer,
            preprocess_config=rgs.model_config[model_class],
            required_features=model.INPUT_COLS, 
        )

        evaluate(
            model=model,
            dataset=test_dataset,
            args=args,
        )


if __name__ == "__main__":

    log_dir = Path("../") / "log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = log_dir / str(uuid.uuid1())
    print("Logging to", log_path)

    logging.basicConfig(
        handlers=[logging.FileHandler(log_path, "w+", "utf-8"), logging.StreamHandler()], 
        format="%(message)s", 
        level=logging.INFO
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--config_dir", type=str, default="../config")
    args = parser.parse_args()

    config_dir = Path(config_dir)
    args.config_dir = config_dir
    args.data_config = load_yaml(config_dir / "data.yaml")
    args.model_config = load_yaml(config_dir / "model.yaml")
    args.train_config = load_yaml(config_dir / "train.yaml")
    args.eval_config = load_yaml(config_dir / "eval.yaml")

    run_bert_unit_content(args=args)
