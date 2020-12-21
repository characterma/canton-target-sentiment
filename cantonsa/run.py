# coding=utf-8
import logging
from pathlib import Path
import os, shutil, sys
import uuid

if __name__=="__main__":
    log_dir = Path(os.environ["CANTON_SA_DIR"]) / "log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = log_dir / str(uuid.uuid1())

    print("Logging to", log_path)
    handlers = [logging.FileHandler(log_path, "w+", "utf-8"), logging.StreamHandler()]
    logging.basicConfig(handlers=handlers, format="%(message)s", level=logging.INFO)

import argparse
import pandas as pd
from cantonsa.dataset import TDSADataset
from cantonsa.trainer import Trainer
from cantonsa.evaluater import Evaluater
from cantonsa.transformers_utils import PretrainedLM
from cantonsa.timer import Timer
from cantonsa.utils import (
    init_logger,
    set_seed,
    load_yaml,
    save_yaml,
    get_label_map,
    generate_grid_search_params,
    apply_grid_search_params,
)
from cantonsa.tokenizer import get_tokenizer
from cantonsa.constants import MODEL_EMB_TYPE
from cantonsa.models import *
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
        state_path=None
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
    overwriting_config_file=None,
    log_path=None, 
    device="cpu",
):
    init_logger()

    base_dir = Path(os.environ["CANTON_SA_DIR"])
    config_dir = base_dir / "config"

    if overwriting_config_file:
        overwriting_config = load_yaml(config_dir / overwriting_config_file)
    else:
        overwriting_config = dict()
    eval_config = load_yaml(
        config_dir / eval_config_file,
        overwriting_config=overwriting_config.get("eval", dict()),
    )

    if do_train:
        data_config = load_yaml(
            config_dir / data_config_file,
            overwriting_config=overwriting_config.get("data", dict()),
        )
        train_config = load_yaml(
            config_dir / train_config_file,
            overwriting_config=overwriting_config.get("train", dict()),
        )
        model_config = load_yaml(
            config_dir / model_config_file,
            overwriting_config=overwriting_config.get("model", dict()),
        )
        grid_config = load_yaml(
            config_dir / grid_config_file,
            overwriting_config=overwriting_config.get("grid", dict()),
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
            eval_input_dir = train_config["output_dir"]
            eval_state_path = train_output_dir / eval_config["state_file"]
            eval_output_dir = (
                base_dir / "output" / "eval" / (train_config["output_dir"] + "_eval")
            )
        else:
            eval_input_dir = eval_config["input_dir"]

            if eval_config["use_train_data_config"]:
                data_config = load_yaml(
                    base_dir / "output" / "train" / eval_input_dir / data_config_file, 
                    overwriting_config=overwriting_config.get("data", dict()),
                )
            else:
                data_config = load_yaml(config_dir / data_config_file, 
                                        overwriting_config=overwriting_config.get("data", dict()),)

            train_config = load_yaml(
                base_dir / "output" / "train" / eval_input_dir / train_config_file, 
                overwriting_config=overwriting_config.get("train", dict()),
            )
            model_config = load_yaml(
                base_dir / "output" / "train" / eval_input_dir / model_config_file, 
                overwriting_config=overwriting_config.get("model", dict()),
            )
            grid_config = load_yaml(
                base_dir / "output" / "train" / eval_input_dir / grid_config_file, 
                overwriting_config=overwriting_config.get("grid", dict()),
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
        save_yaml(eval_config, train_output_dir / eval_config_file)

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
    tokenizer = get_tokenizer(source=preprocess_config["tokenizer_source"], name=preprocess_config["tokenizer_name"])
    pretrained_lm = None
    pretrained_word_emb = None
    word2idx = None
    add_special_tokens = True

    if MODEL_EMB_TYPE[model_class] == "BERT":
        pretrained_lm = PretrainedLM(body_config["pretrained_lm"])
        pretrained_lm.resize_token_embeddings(tokenizer=tokenizer)

    elif MODEL_EMB_TYPE[model_class] == "WORD":
        assert "pretrained_word_emb" in body_config
        assert "use_pretrained" in body_config
        use_pretrained = body_config["use_pretrained"]
        add_special_tokens = False
        if use_pretrained:
            word2idx = None
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
            _, emb_dim = pretrained_word_emb.vectors.shape
            pretrained_word_emb.add(["<OOV>"], [[0] * emb_dim])
            vocab = pretrained_word_emb.vocab
            for token in vocab:
                word2idx[token] = vocab[token].index
        else:
            pretrained_word_emb = None

    label_map = get_label_map(dataset_dir / data_config["label_map"])

    model = init_model(model_class=model_class, 
        body_config=body_config,
        num_labels=len(label_map), 
        pretrained_emb=pretrained_word_emb, 
        num_emb=len(word2idx) if word2idx else None, 
        pretrained_lm=pretrained_lm, 
        device=device, 
        state_path=train_state_path if train_state_path else (None if do_train else eval_state_path)
    )

    # load and preprocess data
    if do_train:
        train_dataset = TDSADataset(
            dataset_dir / data_config["train"],
            label_map,
            tokenizer,
            preprocess_config=preprocess_config,
            word2idx=word2idx,
            add_special_tokens=add_special_tokens,
            to_df=True,
            show_statistics=True,
            name="train",
        )

        dev_evaluators = []
        for _, dev_info in data_config["dev"].items():
            dev_name = dev_info["name"]
            dev_file = dev_info["file"]
            dev_dataset = TDSADataset(
                dataset_dir / dev_file,
                label_map,
                tokenizer,
                preprocess_config=preprocess_config,
                word2idx=word2idx,
                add_special_tokens=add_special_tokens,
                to_df=True,
                show_statistics=True,
                name=f"dev_{dev_name}",
            )
            dev_evaluators.append(
                    Evaluater(
                    model=model, 
                    eval_config=eval_config,
                    output_dir=train_output_dir,
                    dataset=dev_dataset,
                    save_preds=True,
                    save_reps=True,
                    return_losses=False, 
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

            test_dataset = TDSADataset(
                dataset_dir / test_file,
                label_map,
                tokenizer,
                preprocess_config=preprocess_config,
                word2idx=word2idx,
                add_special_tokens=add_special_tokens,
                to_df=True,
                show_statistics=True,
                name=f"test_{test_name}",
                timer=timer
            )

            test_evaluator = Evaluater(
                    model=model, 
                    eval_config=eval_config,
                    output_dir=eval_output_dir,
                    dataset=test_dataset,
                    save_preds=True,
                    save_reps=True,
                    return_losses=False, 
                    timer=timer, 
                    device=device,
                    )

            test_evaluator.evaluate()
            test_evaluator.save_scores()
            timer.save_timer()

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
    parser.add_argument("--overwriting_config", type=str, default="")
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
        overwriting_config_file=args.overwriting_config,
        log_path=log_path, 
        device=args.device,
    )
