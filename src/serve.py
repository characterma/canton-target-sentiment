from inferlight import LightWrapper, BaseInferLightWorker
import time
from sanic import Sanic
from sanic.response import json as json_response
import random
import logging
import argparse

import numpy as np
from scipy.special import softmax
import torch
from torch import nn

from model import get_model
from tokenizer import get_tokenizer
from utils import load_config 
from label import get_label_to_id
from dataset import get_feature_class
from dataset.utils import get_model_inputs


logging.basicConfig(level=logging.INFO)


def parse_request(request):
    data_dict = request.json

    return data_dict


class Worker(BaseInferLightWorker):
    # https://github.com/thuwyh/InferLight/blob/fb1d59703f45d7b9ada4e981fa1dcf306be873b4/inferlight/worker.py#L11

    def load_model(self, args):
        self.model = get_model(args=args)
        self.device = args.device 
        self.args = args
        return 

    def build_batch(self, data):
        batch = dict()
        for col in data[0]:
            batch[col] = torch.stack([x[col] for x in data], dim=0).to(self.device)

        return batch 

    @torch.no_grad()
    def inference(self, batch):
        output = self.model(**batch) 
        output = [self.args.label_to_id_inv[i] for i in output[1]]
        return output

        

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="../config/")
    parser.add_argument("--api_name", type=str, default="predict")
    args = parser.parse_args()

    args = load_config(args)
    tokenizer = get_tokenizer(args=args)
    feature_class = get_feature_class(args)

    label_to_id, label_to_id_inv = get_label_to_id(tokenizer, args)
    args.label_to_id = label_to_id
    args.label_to_id_inv = label_to_id_inv

    app = Sanic('test')
    model = LightWrapper(Worker, args, batch_size=16, max_delay=0.05)
    required_features = get_model_inputs(args=args)


    @app.post(f'/{args.api_name}')
    async def predict(request):
        data_dict = parse_request(request)
        feature = feature_class(
            data_dict=data_dict,
            tokenizer=tokenizer,
            args=args,
            diagnosis=False
        )
        feature_dict = feature.feature_dict
        response = await model.predict(feature_dict)
        if not response.succeed():
            return json_response({'output':None, 'status':'failed'})
        return json_response({'output': response.result})

    app.run(host="0.0.0.0", port=8080, debug=True)