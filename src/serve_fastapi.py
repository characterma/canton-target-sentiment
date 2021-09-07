
import argparse
import re
import traceback
import torch
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI, HTTPException

from model import get_model
from tokenizer import get_tokenizer
from utils import load_config
from label import get_label_to_id
from dataset import get_feature_class
from dataset.utils import get_model_inputs


APP_NAME = "wbi_org_sentiment"


class ModelRunner(object):
    def __init__(self, args):
        self.model = get_model(args=args)
        self.device = args.device
        self.label_to_id_inv = args.label_to_id_inv

    @torch.no_grad()
    def predict(self, feature_dict):
        batch = dict()
        for col in feature_dict:
            batch[col] = feature_dict[col].unsqueeze(0).to(self.device)
        output = self.model(**batch)
        sentiment = self.label_to_id_inv[output["prediction"][0]]
        scores = {}
        for i, s in enumerate(output["probabilities"].squeeze(0).tolist()):
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
    organization: str
    source: str
    pub_code: str
    headline : str
    content : str
    target_keywords: list


def format_result(data_dict, sentiment):
    data_dict["sentiment"] = sentiment
    return data_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="../config/")
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()
    device = args.device
    args = load_config(args)
    if device!="":
        args.device = device
    tokenizer = get_tokenizer(args=args)
    feature_class = get_feature_class(args)

    args.label_to_id, args.label_to_id_inv = get_label_to_id(tokenizer, args)

    app = FastAPI()
    runner = ModelRunner(args)

    @app.post(f'/{APP_NAME}', status_code=200)
    async def run_api(json_dict:Article):
        data_dict = dict(json_dict)
        raw = data_dict.copy()
        target_locs_hl, target_locs_ct = find_target_locs(
            headline=data_dict['headline'], 
            content=data_dict['content'], 
            target_keywords=data_dict['target_keywords']
        )
        data_dict['target_locs_hl'] = target_locs_hl
        data_dict['target_locs_ct'] = target_locs_ct

        feature_dict = feature_class(
            data_dict=data_dict, tokenizer=tokenizer, args=args, diagnosis=False, padding=False
        ).feature_dict
        if feature_dict is None:
            raise HTTPException(status_code=422, detail=f"Target not found or exceeding maximum length ({args.model_config['max_length']}).")
        sentiment, scores, score = runner.predict(feature_dict=feature_dict)
        return {"sentiment": sentiment, "scores": scores, "score": score}

    uvicorn.run(app, host='0.0.0.0', port=8080)