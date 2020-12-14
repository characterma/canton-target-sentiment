import logging
from pathlib import Path
import os, shutil
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
from cantonsa.pipeline import Pipeline
from cantonsa.transformers_utils import PretrainedLM
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
import json
from gensim.models import KeyedVectors

logger = logging.getLogger(__name__)


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
    # log_handlers = []

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
                    base_dir / "output" / "train" / eval_input_dir / data_config_file
                )
            else:
                data_config = load_yaml(config_dir / data_config_file)
            train_config = load_yaml(
                base_dir / "output" / "train" / eval_input_dir / train_config_file
            )
            model_config = load_yaml(
                base_dir / "output" / "train" / eval_input_dir / model_config_file
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
            # print("*******", pretrained_word_emb.vectors.shape)
            pretrained_word_emb.add(["<OOV>"], [[0] * emb_dim])
            vocab = pretrained_word_emb.vocab
            for token in vocab:
                word2idx[token] = vocab[token].index
            # print("*******", len(word2idx))
        else:
            pretrained_word_emb = None

    label_map = get_label_map(dataset_dir / data_config["label_map"])

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
        dev_dataset = dict()
        for dev_idx, dev_info in data_config["dev"].items():
            dev_name = dev_info["name"]
            dev_file = dev_info["file"]
            dev_dataset[dev_name] = TDSADataset(
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
    else:
        train_dataset = None
        dev_dataset = None

    if do_eval:
        test_dataset = dict()
        for ts_idx, ts_info in data_config["test"].items():
            ts_name = ts_info["name"]
            ts_file = ts_info["file"]
            test_dataset[ts_name] = TDSADataset(
                dataset_dir / ts_file,
                label_map,
                tokenizer,
                preprocess_config=preprocess_config,
                word2idx=word2idx,
                add_special_tokens=add_special_tokens,
                to_df=True,
                show_statistics=True,
                name=f"test_{ts_name}",
            )
    else:
        test_dataset = None

    train_id = 0
    if do_train and train_config["grid_search"]:
        logger.info("***** Starting grid search *****")
        param_comb = generate_grid_search_params(grid_config[model_class])
        logger.info("  Number of combinations = '%s'", str(len(param_comb)))
        all_train_results = []
        global_best_scores = None
        for params in param_comb:
            train_results = {}
            body_config, optim_config = apply_grid_search_params(
                params, body_config, optim_config
            )
            pipeline = Pipeline(
                train_id=train_id,
                train_config=train_config,
                eval_config=eval_config,
                body_config=body_config,
                optim_config=optim_config,
                data_config=data_config,
                pretrained_lm=pretrained_lm,
                pretrained_word_emb=pretrained_word_emb,
                label_map=label_map,
                output_dir=train_output_dir,
                train_dataset=train_dataset,
                dev_dataset=dev_dataset,
                test_dataset=test_dataset,
                state_path=train_state_path,
                save_eval_scores=False,
                save_eval_details=False,
                global_best_scores=global_best_scores,
                num_emb=len(word2idx),
                device=device,
            )
            best_epoch, best_step, best_scores, state_filename = pipeline.train()
            if (
                global_best_scores is None
                or global_best_scores["macro_f1"] < best_scores["macro_f1"]
            ):
                global_best_scores = best_scores
            train_results["train_id"] = train_id
            train_results["params"] = json.dumps(params)
            train_results["best_epoch"] = best_epoch
            train_results["best_step"] = best_step
            train_results["state_filename"] = state_filename
            for sc, v in best_scores.items():
                train_results[f"best_{sc}"] = v
            train_id += 1
            all_train_results.append(train_results)
        all_train_results = pd.DataFrame(data=all_train_results)
        all_train_results.to_csv(train_output_dir / "grid_search_results.csv")
        # remove suboptimal model states
        # all_train_results = all_train_results.sort_values("best_macro_f1", ascending=False)
        files_to_rm = all_train_results[
            all_train_results["best_macro_f1"]
            != all_train_results["best_macro_f1"].max()
        ]["state_filename"].tolist()
        for f in files_to_rm:
            if f is not None:
                os.remove(train_output_dir / f)
        if log_path is not None:
            shutil.copy(log_path, train_output_dir)

    elif do_train:
        # print(body_config)
        # print(optim_config)
        pipeline = Pipeline(
            train_id=train_id,
            train_config=train_config,
            eval_config=eval_config,
            body_config=body_config,
            optim_config=optim_config,
            data_config=data_config,
            pretrained_lm=pretrained_lm,
            pretrained_word_emb=pretrained_word_emb,
            label_map=label_map,
            output_dir=train_output_dir,
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            test_dataset=test_dataset,
            state_path=train_state_path,
            save_eval_scores=True,
            save_eval_details=True,
            global_best_scores=None,
            num_emb=len(word2idx) if word2idx is not None else 0,
            device=device,
        )
        pipeline.train()
        if log_path is not None:
            shutil.copy(log_path, train_output_dir)

    if do_eval:
        pipeline = Pipeline(
            train_id=train_id,
            train_config=train_config,
            eval_config=eval_config,
            body_config=body_config,
            optim_config=optim_config,
            data_config=data_config,
            pretrained_lm=pretrained_lm,
            pretrained_word_emb=pretrained_word_emb,
            label_map=label_map,
            output_dir=eval_output_dir,
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            test_dataset=test_dataset,
            state_path=eval_state_path,
            save_eval_scores=True,
            save_eval_details=True,
            global_best_scores=None,
            num_emb=len(word2idx) if word2idx is not None else 0,
            device=device,
        )
        pipeline.eval()
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
