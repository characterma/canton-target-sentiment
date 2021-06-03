import sys
import logging
import torch
from pathlib import Path
from model.TDBERT import TDBERT
from model.TGSAN import TGSAN
from model.TGSAN2 import TGSAN2 


logger = logging.getLogger(__name__)


def get_model(args, model_path=None):
    logger.info("***** Initializing model *****")
    logger.info("  Model class = %s",  args.train_config['model_class'])

    Model = getattr(sys.modules[__name__], args.train_config['model_class'])
    if model_path is None :
        model = Model(args=args)
    else:
        model = torch.load(model_path)
    return model


def get_model_type(args):
    model_class = args.train_config['model_class']
    Model = getattr(sys.modules[__name__], model_class)
    return Model.MODEL_TYPE
