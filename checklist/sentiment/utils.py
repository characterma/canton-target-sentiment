import re
import os
import json
import numpy as np
import pandas as pd
from opencc import OpenCC
from collections import OrderedDict
from pathlib import Path

from nlp_pipeline.utils import (
    set_seed,
    load_yaml,
    set_log_path,
    save_config,
    log_args,
    get_args,
)


def _convert_t2s(convert_func, data):
    if isinstance(data, str):
        return convert_func(data)
    elif isinstance(data, list):
        return [convert_func(d) for d in data]
    else:
        return data


def _concat_target_senti_text(record):
    indices = record["text_subjs"]["text_idxs"]
    text = record["content"]

    content = []
    for index_pair in indices:
        start_index = index_pair[0]
        end_index = index_pair[1]
        target_text = text[start_index: end_index]
        content.append(target_text)

    return " ".join(content)


def load_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def load_checklist_data(fn):
    cc = OpenCC('t2s')
    with open(fn, "r", encoding='utf-8') as f:
        if fn.suffix == ".json":
            data = json.load(f)
            assert len(data) > 0
            if isinstance(data, list) and "content" in data[0]:
                is_target_senti = True if "text_subjs" in data[0] else False

                # for w in data:
                #     if is_target_senti:
                #         pass
                #         # w["content"] = _convert_t2s(cc.convert, _concat_target_senti_text(w))
                #     else:
                #         w["content"] = _convert_t2s(cc.convert, w["content"])
            elif isinstance(data, dict):
                data = {_convert_t2s(cc.convert, k): _convert_t2s(cc.convert, v) for k, v in data.items()}

        else:
            data = f.read().split("\n")
            data = [cc.convert(w) for w in data]

    return data


def load_checklist_config(args):
    args.config_dir = Path(args.config_dir)
    config = load_yaml(args.config_dir / "checklist.yaml")

    # output
    args.output_dir = Path(config["output_dir"])
    args.output_config_dir = args.output_dir / "config/"
    args.output_detail_dir = args.output_dir / "predictions"
    args.device = config["device"]
    args.seed = int(config["seed"])
    
    # model
    args.model_type = str(config["model"]["model_type"]).strip()

    if "api" in args.model_type:
        args.model_path = str(config["model"]["model_path"])
    else:
        args.model_dir = Path(config["model"]["model_path"])

    # data resources
    args.resources = config["resources"]

    # testcases
    args.tc_mft = config["testcases"].get("MFT", None)
    args.tc_inv = config["testcases"].get("INV", None)
    args.tc_dir = config["testcases"].get("DIR", None)

    os.makedirs(args.output_dir, exist_ok=True)

    return args


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
