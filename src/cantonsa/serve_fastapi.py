# coding=utf-8
import json
import uvicorn
from fastapi import FastAPI

from pathlib import Path
import os, sys
import pandas as pd
from cantonsa.transformers_utils import PretrainedLM
from cantonsa.constants import SENTI_ID_MAP_INV
from cantonsa.utils import load_yaml
from cantonsa.tokenizer import get_tokenizer
from cantonsa.models import *
from cantonsa.dataset import TargetDependentExample
import traceback
import torch
from pydantic import BaseModel

if len(sys.argv) > 1:
    device = int(sys.argv[1])
else:
    device = 1

def init_model(
        model_class, 
        body_config, 
        num_labels, 
        pretrained_emb, 
        num_emb, 
        pretrained_lm, 
        device, 
        state_path=None
    ):

    MODEL = getattr(sys.modules[__name__], model_class)
    model = MODEL(
        model_config=body_config,
        num_labels=num_labels,
        pretrained_emb=pretrained_emb,
        num_emb=num_emb,
        pretrained_lm=pretrained_lm,
        device=device,
    )
    if state_path is not None:
        model.load_state(state_path)
    return model

base_dir = Path(os.environ["CANTON_SA_DIR"])
config_dir = base_dir / "config"
deploy_config = load_yaml(config_dir / "deploy.yaml")

model_dir = base_dir / "models" / deploy_config["model_dir"]

model_config = load_yaml(model_dir / "model.yaml")
train_config = load_yaml(model_dir / "train.yaml")

state_path = model_dir / deploy_config["state_file"]

model_class = train_config["model_class"]
preprocess_config = model_config[model_class]["preprocess"]
body_config = model_config[model_class]["body"]
optim_config = model_config[model_class]["optim"]

# load pretrained language model
tokenizer = get_tokenizer(source=preprocess_config["tokenizer_source"], name=preprocess_config["tokenizer_name"])
pretrained_lm = None
pretrained_word_emb = None
word2idx = None
add_special_tokens = True

pretrained_lm = PretrainedLM(body_config["pretrained_lm"])
pretrained_lm.resize_token_embeddings(tokenizer=tokenizer)

model = init_model(model_class=model_class, 
    body_config=body_config,
    num_labels=3, 
    pretrained_emb=pretrained_word_emb, 
    num_emb=len(word2idx) if word2idx else None, 
    pretrained_lm=pretrained_lm, 
    device=device, 
    state_path=state_path
)

model.eval()

# fastapi app
app = FastAPI()

# fastapi user define class
class Item(BaseModel):
    content : str
    start_ind  : int
    end_ind: int

# # API caller - test page
# @app.get('/')
# def index():
#     return "Connect Success"

# API caller - single predict for PI-HK
@app.post('/target_sentiment')
async def pi_single(json_dict:Item):

    # await json_dict
    content = json_dict.content
    start_ind = json_dict.start_ind
    end_ind = json_dict.end_ind

    status = 200

    e = TargetDependentExample(
        raw_text=content,
        raw_start_idx=start_ind,
        raw_end_idx=end_ind,
        tokenizer=tokenizer, 
        label=None,
        preprocess_config=preprocess_config,
        required_features=model.INPUT_COLS
    )

    if e.feature_succeeded and e.preprocess_succeeded:
        with torch.no_grad():
            input = dict()
            for col in model.INPUT_COLS:
                if col in e.features:
                    input[col] = torch.unsqueeze(e.features[col], dim=0).to(device)

            output = model(**input, return_reps=False)
            prediction = torch.argmax(output[1], dim=1).detach().cpu().numpy()
            prediction = SENTI_ID_MAP_INV[prediction[0]]
    else:
        prediction = "neutral"

    res = {
        'status': status,
        'data': prediction
    }

    # logger.info(f'Input headline : {headline}')
    # logger.info(f'Output score : {pi_score:.4f}')
    return res


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)
