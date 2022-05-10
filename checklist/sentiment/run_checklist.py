import argparse

from checklist.pred_wrapper import PredictorWrapper
from checklist_wrapper import SentimentCheckList
from model_wrapper import get_predict_func
from utils import load_checklist_config
from nlp_pipeline.utils import (
    set_seed,
    set_log_path,
    load_config,
    save_config,
    log_args,
    get_args,
)


def run_checklist(args):
    args = load_checklist_config(args)
    set_log_path(args.output_dir)

    set_seed(args.seed)
    args.checklist_config_dir = args.config_dir

    if "api" not in args.model_type:
        model_dir = args.model_dir
        output_dir = args.output_dir
        device = args.device
        args.config_dir = model_dir
        args = load_config(args)
        args.model_dir = model_dir
        args.output_dir = output_dir
        args.device = device

    predict_func = get_predict_func(args)
    checklist = SentimentCheckList(args)

    pred_wrapper = PredictorWrapper.wrap_predict(predict_func)
    checklist.run_test(pred_wrapper)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="../../config/")
    args = parser.parse_args()

    run_checklist(args)
