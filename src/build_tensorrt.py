#!/usr/bin/env python
# coding: utf-8
# %%

# %%





# %%


import argparse
import logging
import json
import torch
import sys

import torch_tensorrt

sys.path.append("../src/")
from model import get_model, get_onnx_session
from tokenizer import get_tokenizer
from utils import load_config, set_log_path, get_args
from label import get_label_to_id
from dataset import get_feature_class
from dataset.utils import get_model_inputs


logger = logging.getLogger(__name__)


def build_tensorrt(args):
    args = load_config(args=args)
    args.device = "cuda"
    set_log_path(args.output_dir)
    tokenizer = get_tokenizer(args=args)
    feature_class = get_feature_class(args)

    label_to_id, label_to_id_inv = get_label_to_id(tokenizer, args)
    args.label_to_id = label_to_id
    args.label_to_id_inv = label_to_id_inv

    data_dict = json.load(open(args.data_dir / args.data_config['test'], "r"))[0]
    feature = feature_class(
        data_dict=data_dict, tokenizer=tokenizer, args=args, diagnosis=False
    )

    feature_dict = feature.feature_dict
    model = get_model(args=args)

    batch = dict()
    for col in feature_dict:
        batch[col] = torch.stack([feature_dict[col]], dim=0).to(torch.int32).to(args.device)
        print(batch[col].device)
    if 'label' in batch:
        del batch['label']

    x = tuple([batch[col].squeeze(-1) for col in batch])
    traced_model = torch.jit.trace(model, x)
    logger.info("***** Build traced model succeeded. *****")

    tensorrt_inputs = []
    for col in batch:
        tensorrt_inputs.append(
            torch_tensorrt.Input(tuple(batch[col].squeeze(-1).shape), dtype=torch.int32)
        )


    trt_model_fp32 = torch_tensorrt.compile(traced_model, **{
        "inputs": tensorrt_inputs,
        "enabled_precisions": {torch.float32}, # Run with FP32
        "workspace_size": 1 << 1000
    })

    logger.info("***** Build tensorrt fp32 succeeded. *****")

    trt_model_fp16 = torch_tensorrt.compile(traced_model, **{
        "inputs": tensorrt_inputs,
        "enabled_precisions": {torch.float16}, # Run with FP32
        "workspace_size": 1 << 1000
    })

    logger.info("***** Build tensorrt fp16 succeeded. *****")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="../output/wbi/org_per_bert_avg_20210925_all_ext2/model")
    args = parser.parse_args()
    args = load_config(args=args)
    set_log_path(args.output_dir)
    build_tensorrt(args=args)