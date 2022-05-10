import os
import logging
import torch
import importlib
from onnxruntime import ExecutionMode, InferenceSession, SessionOptions


logger = logging.getLogger(__name__)


def get_model(args):
    logger.info("***** Initializing model *****")
    model_class = args.train_config["model_class"]
    logger.info("  Task = %s", args.task)
    logger.info("  Model class = %s", model_class)
    Model = getattr(importlib.import_module(f"nlp_pipeline.model.{args.task}"), model_class)
    model_path = args.model_dir / args.eval_config["model_file"]
    if not os.path.exists(model_path):
        model = Model(args=args)
    else:
        logger.info("  Model path = %s", model_path)
        model = Model(args=args)
        model.load_state_dict(torch.load(model_path, map_location=torch.device(args.device)))
    model.return_tensors = None
    return model


def get_onnx_session(args):
    logger.info("***** Initializing onnx model *****")
    model_path = args.model_dir / "model.onnx"
    onnx_session = InferenceSession(str(model_path))
    return onnx_session


def get_jit_traced_model(args):
    logger.info("***** Initializing jit traced model *****")
    model_path = args.model_dir / "traced_model.ts"
    model = torch.jit.load(str(model_path))
    return model
