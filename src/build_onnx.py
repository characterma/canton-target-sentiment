
import logging
import argparse

import torch
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI

from model import get_model, get_onnx_session
from tokenizer import get_tokenizer
from utils import load_config
from label import get_label_to_id
from dataset import get_feature_class
from dataset.utils import get_model_inputs

    
parser = argparse.ArgumentParser()
parser.add_argument("--config_dir", type=str, default="../output/wbi/org_per_bert_avg_20210925_all_ext2/model")
args = parser.parse_args()

args = load_config(args)
args.device = "cpu"
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
  "target_locs_ct": [[21, 24]],
  "entity": "保安局", 
  "pub_code": "wm_stheadlinehk"
}

feature = feature_class(
    data_dict=data_dict, tokenizer=tokenizer, args=args, diagnosis=False
)

feature_dict = feature.feature_dict
model = get_model(args=args)
batch = dict()
for col in feature_dict:
    batch[col] = torch.stack([feature_dict[col]], dim=0).to(args.device)
output = model(**batch)
print(batch['input_ids'].squeeze(-1).size())
x = (batch['input_ids'].squeeze(-1), batch['target_mask'].squeeze(-1), batch['attention_mask'].squeeze(-1), 
batch['token_type_ids'].squeeze(-1))


torch.onnx.export(model,               # model being run
                  args=x,                         # model input (or a tuple for multiple inputs)
                  f=args.model_dir / "model.onnx",   # where to save the model (can be a file or file-like object)
                  do_constant_folding=True,
                  opset_version=11,
                  input_names = ['input_ids', 'target_mask', 'attention_mask', 'token_type_ids'],   # the model's input names
                  output_names = ['probabilities'], # the model's output names
                  dynamic_axes={
                      'input_ids' : {0 : 'batch', 1: 'sequence'}, 
                      'target_mask' : {0 : 'batch', 1: 'sequence'}, 
                      'attention_mask' : {0 : 'batch', 1: 'sequence'}, 
                      'token_type_ids' : {0 : 'batch', 1: 'sequence'}, 
                      'probabilities' : {0 : 'batch', 1: 'class'}})


session = get_onnx_session(args=args)
batch = dict()
for col in feature_dict:
    batch[col] = feature_dict[col].unsqueeze(0).numpy()
output = session.run(None, input_feed=batch)
print(output)
print("Build onnx succeeded.")