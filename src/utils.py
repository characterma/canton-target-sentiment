# coding=utf-8
import logging
import numpy as np
import os
import pandas as pd
import random
import shutil
import torch
import yaml
import pickle
from pathlib import Path
from collections import namedtuple


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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(file_path):
    with open(file_path, "r") as f:
        try:
            data = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as e:
            print(e)
    return data


def set_log_path(log_dir):
    logging.basicConfig(
        handlers=[
            logging.FileHandler(log_dir / "log", "w+", "utf-8"),
            logging.StreamHandler(),
        ],
        format="%(asctime)s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_args(config_dir):
    args = namedtuple("args", ("config_dir",))
    args.config_dir = Path(config_dir)
    return args


def load_config(args, is_deployment=False):
    config_dir = Path(args.config_dir)
    print(config_dir)
    print(os.listdir(config_dir))
    run_config = load_yaml(config_dir / "run.yaml")
    args.run_config = run_config
    args.data_config = run_config["data"]
    args.eval_config = run_config["eval"]
    args.device = run_config["device"]
    args.task = run_config["task"]
    args.train_config = run_config["train"]
    args.kd_config = run_config["train"].get("kd", {"use_kd": False})
    args.prepro_config = run_config["text_prepro"]
    args.explain_config = run_config.get("explanation", {})
    model_config = load_yaml("../config/model.yaml")
    model_class = args.train_config["model_class"]
    args.model_config = model_config[model_class]
    args.model_config.update(run_config["model_params"])

    args.config_dir = config_dir
    if not is_deployment:
        output_dir = Path(args.data_config["output_dir"])
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
        args.tensorboard_dir = Path(output_dir / "logs")
        if not os.path.exists(args.tensorboard_dir):
            os.makedirs(args.tensorboard_dir)
    else:
        args.output_dir = config_dir
        args.model_dir = config_dir
    return args


def save_config(args):
    shutil.copy(args.config_dir / "run.yaml", args.model_dir / "run.yaml")
    shutil.copy("../config/model.yaml", args.model_dir / "model.yaml")


def combine_and_save_metrics(metrics, args):
    metrics = [m for m in metrics if m is not None]
    metrics_df = pd.DataFrame(data=metrics)
    filename = "result_test_only.csv" if args.test_only else "result.csv"
    metrics_df.to_csv(args.result_dir / filename, index=False)


def combine_and_save_statistics(datasets, args):
    datasets = [ds for ds in datasets if ds is not None]
    if hasattr(datasets[0], "diagnosis_df"):
        diagnosis_df = pd.concat([ds.diagnosis_df for ds in datasets])
        try:
            filename = (
                "diagnosis_test_only.xlsx" if args.test_only else "diagnosis.xlsx"
            )

            writer = pd.ExcelWriter(args.result_dir / filename, engine='xlsxwriter')
            diagnosis_df.to_excel(writer, index=False)
            writer.save()

            filename = (
                "diagnosis_test_only.pkl" if args.test_only else "diagnosis.pkl"
            )
            with open(args.result_dir / filename, "wb") as f:
                pickle.dump(diagnosis_df, f)
        except Exception as e:
            print("Failed saving diagnosis_df to excel.", e)
            filename = "diagnosis_test_only.pkl" if args.test_only else "diagnosis.pkl"
            with open(args.result_dir / filename, "wb") as f:
                pickle.dump(diagnosis_df, f)

    if hasattr(datasets[0], "get_data_analysis"):
        statistics_df = pd.DataFrame(data=[ds.get_data_analysis() for ds in datasets])
        try:
            filename = (
                "statistics_test_only.csv" if args.test_only else "statistics.csv"
            )
            statistics_df.to_csv(args.result_dir / filename, index=False)
        except Exception as e:
            print(e)
            filename = "statistics_test_only.pkl" if args.test_only else "statistics.pkl"
            with open(args.result_dir / filename, "wb") as f:
                pickle.dump(statistics_df, f)
