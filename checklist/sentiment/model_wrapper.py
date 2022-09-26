import torch
from torch.utils.data import DataLoader
from nlp_pipeline.dataset import get_dataset
from nlp_pipeline.label import get_label_to_id
from nlp_pipeline.model import get_model, get_onnx_session, get_jit_traced_model
from nlp_pipeline.tokenizer import get_tokenizer
from nlp_pipeline.utils import (
    set_seed,
    set_log_path,
    load_config,
    save_config,
    log_args,
    get_args,
)
from nlp_pipeline.trainer import prediction_step
from utils import load_checklist_config
from pathlib import Path
from tqdm import tqdm
import numpy as np
import logging
import argparse
import requests
import json


logger = logging.getLogger(__name__)


def get_predict_func(args):
    model_type = args.model_type
    if model_type in ["pytorch", "onnx", "jit"]:
        wrapper = SentiModelRunner(args)

    elif model_type == "doc-senti-api":
        wrapper = DocSentiAPICaller(args)

    elif model_type == "target-senti-api":
        wrapper = SubjSentiAPICaller(args)

    else:
        raise AssertionError("model type unknown!")

    return wrapper.predict


class SentiModelRunner():
    def __init__(self, args):
        self.model_type = args.model_type
        self.device = args.device

        tokenizer = get_tokenizer(args=args)
        label_to_id, label_to_id_inv = get_label_to_id(tokenizer, args)
        args.label_to_id = label_to_id
        args.label_to_id_inv = label_to_id_inv

        if self.model_type == "onnx":
            model = get_onnx_session(args=args)
        elif self.model_type == "jit":
            model = get_jit_traced_model(args=args)
            model.eval()
        else:
            model = get_model(args=args)
            model.eval()

        self.model = model
        self.tokenizer = tokenizer

        self.args = args

    @torch.no_grad()
    def predict(self, inputs):

        # preprocess data
        if isinstance(inputs[0], str):
            raw_data = [{"content": text} for text in inputs]
        else:
            raw_data = inputs

        test_dataset = get_dataset(dataset="test", tokenizer=self.tokenizer, args=self.args, raw_data=raw_data)

        # batch data
        dataloader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=self.args.eval_config["batch_size"],
        )

        # predict
        predictions = []
        for batch in tqdm(dataloader, desc="Inferencing"):

            if self.model_type == "onnx":
                batch = {k: v.numpy() for k, v in batch.items() if k != "label"}
                results = self.model.run(None, batch)
                logits = results[0]
                label_ids = np.argmax(logits, 1)
                labels = [self.args.label_to_id_inv[y] for y in label_ids]
                predictions.extend(labels)

            elif self.model_type == "jit":
                batch_tensor = dict()
                for col in batch:
                    if col != "label":
                        batch_tensor[col] = batch[col].to(self.device)

                logits = self.model(**batch_tensor)
                label_ids = logits.argmax(1).cpu().tolist()
                labels = [self.args.label_to_id_inv[y] for y in label_ids]
                predictions.extend(labels)

            else:
                results = prediction_step(self.model, batch, self.args)
                predictions.extend(results["prediction"])

        # insert skipped predictions
        test_dataset.insert_skipped_samples(predictions, 0)

        return predictions


class DocSentiAPICaller():
    def __init__(self, args):
        self.api_endpoint = args.model_path
        self.label_mapping = {
            "neutral": 0,
            "positive": 1,
            "negative": -1,
        }

    def _make_request_payload(self, entry):

        if isinstance(entry, str):
            text = entry

        elif isinstance(entry, dict):
            if "text" in entry:
                text = entry["text"]

            elif "content" in entry:
                text = entry["content"]
            else:
                text = ""
                raise AttributeError("content or text key not found in data dict!")
        else:
            text = ""
            raise AssertionError("data type not support!")

        payload = {
            "text": text
        }

        return payload

    def _get_label_from_api(self, entry):

        payload = self._make_request_payload(entry)

        try:
            response = requests.post(self.api_endpoint, json=payload)
            response = json.loads(response.text)
            return response["data"]["label"]
        except Exception as e:
            print(e)
            print(response)
            return "neutral"

    def predict(self, inputs):
        predictions = [self.label_mapping[self._get_label_from_api(entry)] for entry in inputs]
        return predictions


class SubjSentiAPICaller():
    def __init__(self, args):
        self.api_endpoint = args.model_path

    def _get_label_from_target_senti_api(self, entry):

        payload = {
            "docid": entry.get("docid", "202220110233"),
            "text": entry["content"],
            "text_subjs": entry["text_subjs"],
        }

        try:
            response = requests.post(self.api_endpoint, json=payload)
            response = json.loads(response.text)
            return response["result"][0]["detail"][0]["sentiment"]
        except Exception as e:
            print(e)
            print(response)
            return 0

    def predict(self, inputs):
        predictions = [self._get_label_from_target_senti_api(entry) for entry in inputs]
        return predictions
