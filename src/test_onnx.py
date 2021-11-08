from onnxruntime import ExecutionMode, InferenceSession, SessionOptions

import logging
import argparse

import torch
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI

from model import get_model
from tokenizer import get_tokenizer
from utils import load_config
from label import get_label_to_id
from dataset import get_feature_class
from dataset.utils import get_model_inputs

    
parser = argparse.ArgumentParser()
parser.add_argument("--config_dir", type=str, default="../output/wbi/org_bert_avg_20210906_ext_fixed/model")
args = parser.parse_args()

args = load_config(args)
# args.device = "cpu"
tokenizer = get_tokenizer(args=args)
feature_class = get_feature_class(args)

label_to_id, label_to_id_inv = get_label_to_id(tokenizer, args)
args.label_to_id = label_to_id
args.label_to_id_inv = label_to_id_inv

data_dict = {"organization": "保安局",
  "source": "香江望神州",
  "pub_code": "im_youtube_hk",
  "headline": "鄧炳強批612基金「臨解散都要撈油水」 將作調查 不點名批評黎智英是「主腦」",
  "content": "#國安法#\n撲滅罪行委員會8月27日開會，保安局局長鄧炳強在會後見記者",
  "target_locs_hl": [],
  "target_locs_ct": [[21, 24]]
}

feature = feature_class(
    data_dict=data_dict, tokenizer=tokenizer, args=args, diagnosis=False
)

feature_dict = feature.feature_dict

batch = dict()
for col in feature_dict:
    batch[col] = torch.stack([feature_dict[col]], dim=0).to(args.device)

x = (batch['input_ids'].squeeze(-1), batch['target_mask'].squeeze(-1), batch['attention_mask'].squeeze(-1), 
batch['token_type_ids'].squeeze(-1))

b2 = {
    "input_ids": batch['input_ids'].squeeze(-1).numpy(),
    "target_mask": batch['target_mask'].squeeze(-1).numpy(),
    "attention_mask": batch['attention_mask'].squeeze(-1).numpy(),
    "token_type_ids": batch['token_type_ids'].squeeze(-1).numpy()
}
onnx_session = InferenceSession('./test.onnx')
# print(help(onnx_session.run))
output = onnx_session.run(None, input_feed=b2)

print(output)