import os
import logging
import torch
import importlib


logger = logging.getLogger(__name__)


def get_model(args):
    logger.info("***** Initializing model *****")
    model_class = args.train_config["model_class"]
    logger.info("  Task = %s", args.task)
    logger.info("  Model class = %s", model_class)
    Model = getattr(importlib.import_module(f"model.{args.task}"), model_class)
    model_path = args.model_dir / args.eval_config["model_file"]
    if not os.path.exists(model_path):
        model = Model(args=args)
    else:
        logger.info("  Model path = %s", model_path)
        model = torch.load(model_path, map_location=torch.device(args.device))
    return model
