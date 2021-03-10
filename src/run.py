# coding=utf-8
import logging
from pathlib import Path
import os, shutil, sys
import uuid
import numpy as np
import pickle

if __name__ == "__main__":
    log_dir = Path(os.environ["CANTON_SA_DIR"]) / "log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = log_dir / str(uuid.uuid1())

    print("Logging to", log_path)
    handlers = [logging.FileHandler(log_path, "w+", "utf-8"), logging.StreamHandler()]
    logging.basicConfig(handlers=handlers, format="%(message)s", level=logging.INFO)

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
    generate_grid_search_params,
    apply_grid_search_params,
    Timer,
    MODEL_EMB_TYPE,
)
from tokenizer import get_tokenizer
from model import *
import json
from gensim.models import KeyedVectors

logger = logging.getLogger(__name__)


def init_model(
    model_class,
    body_config,
    num_labels,
    pretrained_emb,
    num_emb,
    pretrained_lm,
    device,
    state_path=None,
):

    MODEL = getattr(sys.modules[__name__], model_class)
    model = MODEL(
        model_config=body_config,
        num_labels=num_labels,
        pretrained_emb=pretrained_emb,
        num_emb=num_emb,
        pretrained_lm=pretrained_lm,
        device=device,
    )
    if state_path is not None:
        model.load_state(state_path)
    return model


def run(
    do_train=False,
    do_eval=False,
    data_config_file="data.yaml",
    train_config_file="train.yaml",
    eval_config_file="eval.yaml",
    model_config_file="model.yaml",
    grid_config_file="grid.yaml",
    log_path=None,
    device="cpu",
):
    init_logger()

    base_dir = Path(os.environ["CANTON_SA_DIR"])
    config_dir = base_dir / "config"

    if do_train:
        data_config = load_yaml(
            config_dir / data_config_file,
        )
        train_config = load_yaml(
            config_dir / train_config_file,
        )
        model_config = load_yaml(
            config_dir / model_config_file,
        )
        grid_config = load_yaml(
            config_dir / grid_config_file,
        )
        eval_config = load_yaml(
            config_dir / eval_config_file,
        )
        if train_config["state_file"]:
            train_state_path = (
                base_dir / "output" / "train" / train_config["state_file"]
            )
        else:
            train_state_path = None
        train_output_dir = base_dir / "output" / "train" / train_config["output_dir"]

        if not os.path.exists(train_output_dir):
            os.makedirs(train_output_dir)
    else:
        train_state_path = None

    if do_eval:
        if do_train:
            eval_config = load_yaml(
                config_dir / eval_config_file,
            )
            eval_input_dir = train_config["output_dir"]
            eval_state_path = train_output_dir / eval_config["state_file"]
            eval_output_dir = (
                base_dir / "output" / "eval" / (train_config["output_dir"] + "_eval")
            )
        else:
            eval_config = load_yaml(
                config_dir / eval_config_file,
            )
            eval_input_dir = eval_config["input_dir"]

            if eval_config["use_train_data_config"]:
                data_config = load_yaml(
                    base_dir / "output" / "train" / eval_input_dir / data_config_file,
                )
            else:
                data_config = load_yaml(config_dir / data_config_file)

            train_config = load_yaml(
                base_dir / "output" / "train" / eval_input_dir / train_config_file,
            )
            model_config = load_yaml(
                base_dir / "output" / "train" / eval_input_dir / model_config_file,
            )
            grid_config = load_yaml(
                base_dir / "output" / "train" / eval_input_dir / grid_config_file,
            )

            eval_state_path = (
                base_dir
                / "output"
                / "train"
                / eval_input_dir
                / eval_config["state_file"]
            )
            eval_output_dir = base_dir / "output" / "eval" / eval_config["output_dir"]
        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)

    if do_train:
        save_yaml(model_config, train_output_dir / model_config_file)
        save_yaml(train_config, train_output_dir / train_config_file)
        save_yaml(grid_config, train_output_dir / grid_config_file)
        save_yaml(data_config, train_output_dir / data_config_file)
        # save_yaml(eval_config, train_output_dir / eval_config_file)

    if do_eval:
        save_yaml(model_config, eval_output_dir / model_config_file)
        save_yaml(train_config, eval_output_dir / train_config_file)
        save_yaml(grid_config, eval_output_dir / grid_config_file)
        save_yaml(data_config, eval_output_dir / data_config_file)
        save_yaml(eval_config, eval_output_dir / eval_config_file)

    model_class = train_config["model_class"]
    preprocess_config = model_config[model_class]["preprocess"]
    body_config = model_config[model_class]["body"]
    optim_config = model_config[model_class]["optim"]
    set_seed(train_config["seed"])
    dataset_dir = base_dir / "data" / "datasets" / data_config["dataset"]

    # load pretrained language model
    tokenizer = get_tokenizer(
        source=preprocess_config["tokenizer_source"],
        name=preprocess_config["tokenizer_name"],
    )
    pretrained_lm = None
    pretrained_word_emb = None
    emb_vectors = None
    word2idx = None
    num_emb = None
    add_special_tokens = True
    label_map = get_label_map(dataset_dir / data_config["label_map"])

    if MODEL_EMB_TYPE[model_class] == "BERT":
        pretrained_lm = PretrainedLM(body_config["pretrained_lm"])
        pretrained_lm.resize_token_embeddings(tokenizer=tokenizer)

    elif MODEL_EMB_TYPE[model_class] == "WORD":
        add_special_tokens = False
        if do_train:
            _vocab = TargetDependentDataset(
                dataset_dir / data_config["train"],
                label_map,
                tokenizer,
                preprocess_config=preprocess_config,
                word2idx=None,
                add_special_tokens=False,
                name="train",
                required_features=None,
                get_vocab_only=True,
            ).vocab

            vocab = dict()
            for k, v in _vocab.items():
                if v >= 2:
                    vocab[k] = v

            word2idx = dict()
            word2idx["<PAD>"] = 0
            word2idx["<OOV>"] = 1
            for k in vocab.keys():
                word2idx[k] = len(word2idx)
            num_emb = len(word2idx)

            if body_config["use_pretrained"]:

                word_emb_path = (
                    base_dir
                    / "data"
                    / "word_embeddings"
                    / body_config["pretrained_word_emb"]
                )
                logger.info("***** Loading pretrained word embeddings *****")
                logger.info("  Pretrained word embeddings = '%s'", str(word_emb_path))

                pretrained_word_emb = KeyedVectors.load_word2vec_format(
                    word_emb_path, binary=False
                )

                emb_dim = pretrained_word_emb.vectors.shape[1]

                emb_vectors = np.random.rand(len(word2idx) + len(vocab), emb_dim)
                glove_vocab = pretrained_word_emb.vocab
                glove_vectors = pretrained_word_emb.vectors

                # migrate glove vectors
                for k in vocab.keys():
                    if k in glove_vocab:
                        glove_idx = glove_vocab[k].index
                        emb_vectors[word2idx[k], :] = glove_vectors[glove_idx, :]
                emb_dim = glove_vectors.shape[1]
            else:
                emb_dim = body_config["emb_dim"]

            word2idx_info = {"word2idx": word2idx, "emb_dim": emb_dim}
            pickle.dump(
                word2idx_info, open(train_output_dir / "word2idx_info.pkl", "wb")
            )

        elif do_eval:
            word2idx_info = pickle.load(
                open(
                    base_dir
                    / "output"
                    / "train"
                    / eval_input_dir
                    / "word2idx_info.pkl",
                    "rb",
                )
            )
            word2idx = word2idx_info["word2idx"]
            emb_dim = word2idx_info["emb_dim"]
            emb_vectors = np.random.rand(len(word2idx), emb_dim)

        num_emb = len(word2idx)
        logger.info("***** Word embeddings *****")
        logger.info("  Number of tokens = '%s'", num_emb)
        logger.info("  Embedding dimension = '%s'", emb_dim)

    model = init_model(
        model_class=model_class,
        body_config=body_config,
        num_labels=len(label_map),
        pretrained_emb=emb_vectors,
        num_emb=num_emb,
        pretrained_lm=pretrained_lm,
        device=device,
        state_path=train_state_path
        if train_state_path
        else (None if do_train else eval_state_path),
    )

    # load and preprocess data
    if do_train:
        train_dataset = TargetDependentDataset(
            dataset_dir / data_config["train"],
            label_map,
            tokenizer,
            preprocess_config=preprocess_config,
            word2idx=word2idx,
            add_special_tokens=add_special_tokens,
            name="train",
            required_features=model.INPUT_COLS,
        )

        dev_evaluators = []
        for _, dev_info in data_config["dev"].items():
            dev_name = dev_info["name"]
            dev_file = dev_info["file"]
            dev_dataset = TargetDependentDataset(
                dataset_dir / dev_file,
                label_map,
                tokenizer,
                preprocess_config=preprocess_config,
                word2idx=word2idx,
                add_special_tokens=add_special_tokens,
                name=dev_name,
                required_features=model.INPUT_COLS,
            )
            dev_evaluators.append(
                Evaluater(
                    model=model,
                    eval_config=eval_config,
                    output_dir=train_output_dir,
                    dataset=dev_dataset,
                    no_save=True,
                    device=device,
                )
            )

        trainer = Trainer(
            model=model,
            train_config=train_config,
            optim_config=optim_config,
            output_dir=train_output_dir,
            dataset=train_dataset,
            dev_evaluaters=dev_evaluators,
            device=device,
        )

        trainer.train()

        if log_path is not None:
            shutil.copy(log_path, train_output_dir)

    if do_eval:
        timer = Timer(output_dir=eval_output_dir)
        eval_results = dict()
        test_evaluators = []
        for _, test_info in data_config["test"].items():
            test_name = test_info["name"]
            test_file = test_info["file"]

            test_dataset = TargetDependentDataset(
                dataset_dir / test_file,
                label_map,
                tokenizer,
                preprocess_config=preprocess_config,
                word2idx=word2idx,
                add_special_tokens=add_special_tokens,
                name=test_name,
                timer=timer,
                required_features=model.INPUT_COLS,
            )

            test_evaluator = Evaluater(
                model=model,
                eval_config=eval_config,
                output_dir=eval_output_dir,
                dataset=test_dataset,
                timer=timer,
                device=device,
            )

            test_evaluator.evaluate()

        if log_path is not None:
            shutil.copy(log_path, eval_output_dir)


if __name__ == "__main__":
    """
    python run.py --do_train
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the test set."
    )
    parser.add_argument("--data_config", type=str, default="data.yaml")
    parser.add_argument("--model_config", type=str, default="model.yaml")
    parser.add_argument("--train_config", type=str, default="train.yaml")
    parser.add_argument("--grid_config", type=str, default="grid.yaml")
    parser.add_argument("--eval_config", type=str, default="eval.yaml")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    run(
        do_train=True if args.do_train else False,
        do_eval=True if args.do_eval else False,
        data_config_file=args.data_config,
        train_config_file=args.train_config,
        eval_config_file=args.eval_config,
        model_config_file=args.model_config,
        grid_config_file=args.grid_config,
        log_path=log_path,
        device=args.device,
    )
