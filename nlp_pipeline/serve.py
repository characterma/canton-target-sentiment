import argparse
import re
import torch
import numpy as np
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

from nlp_pipeline.utils import load_config
from nlp_pipeline.model import get_onnx_session, get_jit_traced_model
from nlp_pipeline.dataset import get_feature
from nlp_pipeline.tokenizer import get_tokenizer
from nlp_pipeline.label import get_label_to_id


def get_feature_func(args):
    tokenizer = get_tokenizer(args=args)
    args.label_to_id, args.label_to_id_inv = get_label_to_id(tokenizer, args)
    def get_feature(data_dict):
        feature_class = get_feature_class(args)
        return feature_class(
            data_dict=data_dict, tokenizer=tokenizer, args=args, diagnosis=False, padding=False
        ).feature_dict
    return get_feature


class OnnxModelRunner(object):
    def __init__(self, args):
        self.session = get_onnx_session(args=args)
        self.device = args.device
        self.label_to_id_inv = args.label_to_id_inv

    @torch.no_grad()
    def predict(self, feature_dict):
        batch = dict()
        for col in feature_dict:
            batch[col] = feature_dict[col].unsqueeze(0).cpu().numpy()
        output = self.session.run(None, input_feed=batch)
        # parse output:
        probabilities = output[0].flatten()
        prediction_id = probabilities.argmax()
        prediction = self.label_to_id_inv[prediction_id]
        scores = {}
        for i, s in enumerate(probabilities.tolist()):
            scores[self.label_to_id_inv[i]] = s
        return prediction, scores


class TracedModelRunner(object):
    def __init__(self, args):
        self.model = get_jit_traced_model(args=args)
        self.device = args.device
        self.label_to_id_inv = args.label_to_id_inv

    @torch.no_grad()
    def predict(self, feature_dict):
        batch = dict()
        for col in feature_dict:
            batch[col] = feature_dict[col].unsqueeze(0).to(self.device)
        output = self.model(**batch)
        # parse output:
        probabilities = output[0].flatten()  # ???
        prediction_id = probabilities.argmax()
        prediction = self.label_to_id_inv[prediction_id]
        scores = {}
        for i, s in enumerate(probabilities.tolist()):
            scores[self.label_to_id_inv[i]] = s
        return prediction, scores





class APIInput(BaseModel):
    ##########################################
    # Define API input here
    ##########################################
    content : str


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="../model/")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--model_type", type=str, default="")
    args = parser.parse_args()
    device = args.device

    args = load_config(args, is_deployment=True)
    if device!="":
        args.device = device


    get_feature = get_feature_func(args=args)


    if args.model_type == "onnx":
        runner = OnnxModelRunner(args)
    elif args.model_type == "traced":
        runner = TracedModelRunner(args)

    app = FastAPI()

    @app.post(f'/{END_PT}', status_code=200)
    async def run_api(json_dict:APIInput):

        data_dict = dict(json_dict)

        ##########################################
        # Add transformation for data_dict here
        ##########################################

        feature_dict = get_feature(data_dict=data_dict)

        if feature_dict is None:
            raise HTTPException(status_code=422, detail="Make feature failed.")

        prediction, scores = runner.predict(feature_dict=feature_dict)

        ##########################################
        # Format API output here
        ##########################################
        api_output = {"prediction": prediction, "scores": scores}

        return api_output

    uvicorn.run(app, host='0.0.0.0', port=8080)