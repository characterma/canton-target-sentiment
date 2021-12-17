
import argparse
import re
import traceback
import torch
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI, HTTPException

import numpy as np
from model import get_onnx_session
from tokenizer import get_tokenizer
from utils import load_config
from label import get_label_to_id
from dataset import get_feature_class
from dataset.utils import get_model_inputs


END_PT = "predict"
TH_POS = TH_NEG = 0.0072


class ModelRunner(object):
    def __init__(self, args):
        self.session = get_onnx_session(args=args)
        self.device = args.device
        self.label_to_id_inv = args.label_to_id_inv

    @torch.no_grad()
    def predict(self, feature_dict):
        batch = dict()
        for col in feature_dict:
            batch[col] = feature_dict[col].unsqueeze(0).numpy()
        output = self.session.run(None, input_feed=batch)
        probabilities = output[0].flatten()
        sentiment_id = probabilities.argmax()
        sentiment = self.label_to_id_inv[sentiment_id]
        scores = {}
        for i, s in enumerate(probabilities.tolist()):
            scores[self.label_to_id_inv[i]] = s
        score = scores[sentiment]
        return sentiment, scores, score


def find_target_locs(headline, content, target_keywords):
    target_locs_hl = []
    target_locs_ct = []
    pattern = re.compile("|".join(target_keywords), re.IGNORECASE)
    if headline:
        for match in re.finditer(pattern, headline):
            st_idx = match.start()
            ed_idx = match.end()
            grp = match.group()
            target_locs_hl.append([st_idx, ed_idx])
    if content:
        for match in re.finditer(pattern, content):
            st_idx = match.start()
            ed_idx = match.end()
            grp = match.group()
            target_locs_ct.append([st_idx, ed_idx])
    return target_locs_hl, target_locs_ct


class Article(BaseModel):
    entity: str
    source: str
    pub_code: str
    headline : str
    content : str
    extended_target_keywords: list


def format_result(data_dict, sentiment):
    data_dict["sentiment"] = sentiment
    return data_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="../model/")
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()
    device = args.device
    args = load_config(args, is_deployment=True)
    if device!="":
        args.device = device
    tokenizer = get_tokenizer(args=args)
    feature_class = get_feature_class(args)

    args.label_to_id, args.label_to_id_inv = get_label_to_id(tokenizer, args)

    app = FastAPI()
    runner = ModelRunner(args)

    @app.post(f'/{END_PT}', status_code=200)
    async def run_api(json_dict:Article):
        data_dict = dict(json_dict)
        raw = data_dict.copy()
        target_locs_hl, target_locs_ct = find_target_locs(
            headline=data_dict['headline'], 
            content=data_dict['content'], 
            target_keywords=data_dict['extended_target_keywords']
        )
        data_dict['target_locs_hl'] = target_locs_hl
        data_dict['target_locs_ct'] = target_locs_ct

        feature_dict = feature_class(
            data_dict=data_dict, tokenizer=tokenizer, args=args, diagnosis=False, padding=False
        ).feature_dict

        if feature_dict is None:
            raise HTTPException(status_code=422, detail=f"Target not found or exceeding maximum length ({args.model_config['max_length']}).")
        sentiment, scores, score = runner.predict(feature_dict=feature_dict)

        if TH_NEG is not None and scores['negative'] > TH_NEG:
            need_pr = True 
        elif TH_POS is not None and scores['positive'] > TH_NEG:
            need_pr = True 
        else:
            need_pr = False
            
        return {"sentiment": sentiment, "scores": scores, "score": score, "need_pr": need_pr}

    uvicorn.run(app, host='0.0.0.0', port=8080)