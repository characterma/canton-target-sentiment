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
from cantonsa.utils import init_logger, load_yaml, save_yaml
from cantonsa.run import run

logger = logging.getLogger(__name__)


def run_multiple(args, log_path=None):
    init_logger()
    base_dir = Path(os.environ["CANTON_SA_DIR"])
    exp_dir = base_dir / "experiments"
    config_dir = base_dir / "config"
    experiment = load_yaml(exp_dir / f"{args.experiment}.yaml")
    for exp_idx, exp_config in experiment.items():
        exp_name = "{0}/{1}".format(args.experiment, exp_config["name"])
        logger.info("***** Running experiment %d: %s *****", exp_idx, exp_name)
        if "train" in exp_config:
            exp_config["train"]["output_dir"] = exp_name
        else:
            exp_config["train"] = {"output_dir": exp_name}
        save_yaml(exp_config, config_dir / "overwrite_tmp.yaml")

        run(
            do_train=True,
            do_eval=True,
            data_config_file="data.yaml",
            train_config_file="train.yaml",
            eval_config_file="eval.yaml",
            model_config_file="model.yaml",
            grid_config_file="grid.yaml",
            overwriting_config_file="overwrite_tmp.yaml",
            log_path=log_path, 
            device=args.device,
        )
    if log_path is not None:
        os.remove(log_path)


if __name__ == "__main__":
    """
    python run_multi.py --experiment='' --device=2
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default=".yaml")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    run_multiple(args, log_path=log_path)
