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

# specialize for mlops tools
import neptune.new as neptune
from neptune.new.types import File

from pathlib import Path, PurePath
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
    log_dir = Path(log_dir)
    logging.basicConfig(
        handlers=[
            logging.FileHandler(log_dir / "log", "a", "utf-8"),
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
    src_dir = Path(PurePath(__file__).parent)
    default_config_dir = src_dir.parent / "config" 
    run_config = load_yaml(config_dir / "run.yaml")
    args.run_config = run_config
    args.data_config = run_config["data"]
    args.eval_config = run_config["eval"]
    args.device = run_config["device"]
    args.task = run_config["task"]
    args.train_config = run_config["train"]
    args.mlops_config = run_config.get("mlops", {})
    args.kd_config = run_config["train"].get("kd", {"use_kd": False}) 
    args.uda_config = run_config["train"].get("uda", {"use_uda": False}) 
    args.prepro_config = run_config["text_prepro"]
    args.explain_config = run_config.get("explanation", {})
    model_config = load_yaml(default_config_dir / "model.yaml")
    model_class = args.train_config["model_class"]
    args.model_config = model_config[model_class]
    args.model_config.update(run_config["model_params"])
    args.config_dir = config_dir
    args.al_config = run_config.get('active_learning', {"run_al_exp": False})

    # specialize for mlops tools
    if args.mlops_config.get("neptune") and args.mlops_config["neptune"]['log']:
            args.mlops_config["neptune"]["run"] = neptune_init(args)

    if not is_deployment:
        output_dir = Path(args.data_config["output_dir"])
        args.output_dir = output_dir
        if not args.output_dir.is_absolute():
            args.output_dir = src_dir / Path(args.output_dir)
            
        args.data_dir = Path(args.data_config["data_dir"])
        if not args.data_dir.is_absolute():
            args.data_dir = src_dir / Path(args.data_dir)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        args.model_dir = args.output_dir / "model"
        args.result_dir = args.output_dir / "result"

        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)
        args.pretrained_emb_path = args.model_config.get("pretrained_emb_path", None)
        args.tensorboard_dir = Path(output_dir / "logs")
        if not os.path.exists(args.tensorboard_dir):
            os.makedirs(args.tensorboard_dir)

        if args.pretrained_emb_path is not None and not Path(args.pretrained_emb_path).is_absolute():
            args.pretrained_emb_path = str(src_dir) + "/" + args.pretrained_emb_path
        print(args.pretrained_emb_path, "**********************************")

    else:
        args.output_dir = Path(config_dir)
        args.model_dir = Path(config_dir)

        if not args.output_dir.is_absolute():
            args.output_dir = src_dir / Path(args.output_dir)
            
        if not args.model_dir.is_absolute():
            args.model_dir = src_dir / Path(args.model_dir)

        if not args.data_dir.is_absolute():
            args.data_dir = src_dir / Path(args.data_dir)
    print(args.model_dir, "*********************************************************")
    return args

def neptune_init(args):
    return neptune.init_run(
                project=args.mlops_config["neptune"]["project"],
                api_token=args.mlops_config["neptune"]["api_token"],
                name=args.mlops_config["neptune"]["name"] 
                        if args.mlops_config["neptune"]["name"] 
                        else args.data_config["output_dir"].split('/')[-1],  # 
                description=args.mlops_config["neptune"]["description"],  # 
                mode=args.mlops_config["neptune"]["mode"],  # 
                tags=args.mlops_config["neptune"]["tags"],  # 
                capture_hardware_metrics=args.mlops_config["neptune"]["capture_hardware_metrics"],  #
    )  # your credentials

def save_config(args):
    default_config_dir = Path(PurePath(__file__).parent).resolve().parent / "config" 
    shutil.copy(args.config_dir / "run.yaml", args.model_dir / "run.yaml")
    shutil.copy(default_config_dir / "model.yaml", args.model_dir / "model.yaml")

def start_mlops_log(args):
    if args.mlops_config.get("neptune"):
        if not args.mlops_config["neptune"]['log']:
            args.mlops_config["neptune"]["run"] = neptune_init(args)
        args.mlops_config["neptune"]["run"]['run_config'] = args.run_config
        args.mlops_config["neptune"]["run"]['model_config'] = args.model_config

def stop_mlops_log(args):
    if args.mlops_config.get("neptune"):
        args.mlops_config["neptune"]["run"].stop()

def combine_and_save_metrics(metrics, args, suffix=None):
    metrics = [m for m in metrics if m is not None]
    metrics_df = pd.DataFrame(data=metrics)
    if suffix is not None:
        filename = f"result_{suffix}.csv"
    else:
        filename = "result_test_only.csv" if args.test_only else "result.csv"
    metrics_df.to_csv(args.result_dir / filename, index=False)
    if args.mlops_config.get("neptune"):
        for metric in metrics:
            args.mlops_config["neptune"]["run"][f"metrics/{metric['dataset']}_metric"] = metric
        if args.task == 'sequence_classification':
            metrics_df.index = metrics_df['dataset']
            args.mlops_config["neptune"]["run"]['plot/classification_report'].upload(File.as_html(metrics_df))

def combine_and_save_statistics(datasets, args, suffix=None):
    datasets = [ds for ds in datasets if ds is not None]
    if hasattr(datasets[0], "diagnosis_df"):
        diagnosis_df = pd.concat([ds.diagnosis_df for ds in datasets])
        try:
            if suffix is not None:
                filename = f"diagnosis_{suffix}.xlsx"
            else:
                filename = (
                    "diagnosis_test_only.xlsx" if args.test_only else "diagnosis.xlsx"
                )
            writer = pd.ExcelWriter(args.result_dir / filename, engine='xlsxwriter')
            diagnosis_df.to_excel(writer, index=False)
            writer.save()
        except Exception as e:
            print(e)

        if suffix is not None:
            filename = f"diagnosis_{suffix}.pkl"
        else:
            filename = (
                "diagnosis_test_only.pkl" if args.test_only else "diagnosis.pkl"
            )
        with open(args.result_dir / filename, "wb") as f:
            pickle.dump(diagnosis_df, f)

    if hasattr(datasets[0], "get_data_analysis"):
        statistics_df = pd.DataFrame(data=[ds.get_data_analysis() for ds in datasets])
        try:
            if suffix is not None:
                filename = f"statistics_{suffix}.csv"
            else:
                filename = (
                    "statistics_test_only.csv" if args.test_only else "statistics.csv"
                )
            statistics_df.to_csv(args.result_dir / filename, index=False)
        except Exception as e:
            print(e)
            if suffix is not None:
                filename = f"statistics_{suffix}.pkl"
            else:
                filename = "statistics_test_only.pkl" if args.test_only else "statistics.pkl"
            with open(args.result_dir / filename, "wb") as f:
                pickle.dump(statistics_df, f)

    if args.task == 'sequence_classification' and args.mlops_config.get("neptune"):
        train_diagnosis_df = diagnosis_df[diagnosis_df['dataset']=='train']
        train_cm = pd.crosstab(train_diagnosis_df['label'], train_diagnosis_df['prediction'], rownames=['label'], colnames=['pred'])
        args.mlops_config["neptune"]["run"]['plot/train_confusion_matrix'].upload(File.as_html(train_cm))
        
        dev_diagnosis_df = diagnosis_df[diagnosis_df['dataset']=='dev']
        dev_cm = pd.crosstab(dev_diagnosis_df['label'], dev_diagnosis_df['prediction'], rownames=['label'], colnames=['pred'])
        args.mlops_config["neptune"]["run"]['plot/dev_confusion_matrix'].upload(File.as_html(dev_cm))

        test_diagnosis_df = diagnosis_df[diagnosis_df['dataset']=='test']
        test_cm = pd.crosstab(test_diagnosis_df['label'], test_diagnosis_df['prediction'], rownames=['label'], colnames=['pred'])
        args.mlops_config["neptune"]["run"]['plot/test_confusion_matrix'].upload(File.as_html(test_cm))
