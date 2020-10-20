import argparse

from data_loader import load_dataset
from trainer import Trainer
from pretrained_model import PretrainedML
from utils import init_logger, set_seed, load_yaml, get_label_map
from pathlib import Path
import os, shutil


def main(args):
    init_logger()

    config_dir = Path(os.environ['CANTON_SA_DIR']) / "config"
    model_config = load_yaml(config_dir / "model.yaml")
    data_config = load_yaml(config_dir / "data.yaml")
    train_config = load_yaml(config_dir / "train.yaml")

    set_seed(train_config['seed'])
    data_dir = Path(os.environ['CANTON_SA_DIR']) / "data" / data_config['data_path']['subfolder']
    model_dir = Path(os.environ['CANTON_SA_DIR']) / "model" / train_config['output']['subfolder']
    if not os.path.exists(model_dir):
        if args.do_train:
            os.makedirs(model_dir)
        elif args.do_eval:
            raise("no model.")

    # load pretrained language model
    pretrained_model = PretrainedML(model_config['lm']['pretrained_model'])
    label_map = get_label_map(data_dir / data_config['data_path']['label_map'])

    # load and preprocess data
    if args.do_train:
        train_dataset, train_details = load_dataset(data_dir / data_config['data_path']['train'], 
                                    label_map, 
                                    pretrained_model.tokenizer, 
                                    model_config['lm']["max_length"], 
                                    details=True)
        dev_dataset, dev_details = load_dataset(data_dir / data_config['data_path']['dev'], 
                                label_map, 
                                pretrained_model.tokenizer, 
                                model_config['lm']["max_length"], 
                                details=True)
    else:
        train_dataset, train_details = None, None
        dev_dataset, dev_details = None, None

    if args.do_eval:
        test_dataset, test_details = load_dataset(data_dir / data_config['data_path']['test'], 
                                    label_map, 
                                    pretrained_model.tokenizer, 
                                    model_config['lm']["max_length"], 
                                    details=True)
    else:
        test_dataset, test_details = None, None

    # start training
    trainer = Trainer(train_config, model_config, data_config, 
                      pretrained_model, label_map, model_dir, 
                      train_dataset=train_dataset, 
                      dev_dataset=dev_dataset, 
                      test_dataset=test_dataset, 
                      train_details=train_details, 
                      dev_details=dev_details, 
                      test_details=test_details)

    if args.do_train:
        trainer.train()
    if args.do_eval:
        trainer.evaluate("test")

    shutil.copy(config_dir / "model.yaml", model_dir)
    shutil.copy(config_dir / "train.yaml", model_dir)
    shutil.copy(config_dir / "data.yaml", model_dir)

if __name__ == "__main__":
    """
    python main.py
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the test set."
    )
    args = parser.parse_args()
    main(args)
