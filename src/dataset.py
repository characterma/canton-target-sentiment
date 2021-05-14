#!/usr/bin/python
# -*- coding: utf-8 -*-
import json
import logging
import os
import sys
import pickle
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from pathlib import Path
from utils import SENTI_ID_MAP, SPEC_TOKEN
from preprocess import TextPreprocessor
from sklearn.utils import resample
from tokenizer import tokenizer_internal
from collections import Counter
from model import *


logger = logging.getLogger(__name__)


def pad_tensor(vec, pad, dim):
    # vec: np.array, pad: int
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size).long()], dim=dim)


class PadCollate:
    def __init__(self, pad_cols, input_cols, dim=0):
        self.dim = dim
        self.pad_cols = pad_cols
        self.input_cols = input_cols

    def pad_collate(self, batch):
        outputs = dict()
        for col in self.input_cols:
            if col in self.pad_cols:
                max_len = max(map(lambda x: x[col].shape[self.dim], batch))
                x_col = list(
                    map(lambda x: pad_tensor(x[col], pad=max_len, dim=self.dim), batch)
                )
                x_col = torch.stack(x_col, dim=0)
            else:
                x_col = torch.stack(
                    list(map(lambda x: torch.tensor(x[col]), batch)), dim=0
                )
            outputs[col] = x_col
        return outputs

    def __call__(self, batch):
        return self.pad_collate(batch)


class TargetDependentExample(object):
    def __init__(
        self,
        data_dict,
        tokenizer,
        prepro_config,
        model_type,
        required_features,
        max_length,
        word_to_idx=None,
    ):
        """
        data_dict: dict
        prepro_config: dict
        model_type: str
        required_features: List
        """
        self.succeeded = True
        raw_text = data_dict['content']
        target_locs = data_dict["target_locs"]
        sentiment = data_dict["sentiment"]
        assert sentiment in ["neutral", "negative", "positive"]

        preprocessor = TextPreprocessor(
            text=raw_text, 
            target_locs=target_locs, 
            steps=prepro_config['steps']
        )
        preprocessed_text = preprocessor.preprocessed_text
        preprocessed_target_locs = preprocessor.preprocessed_target_locs

        # if model_type=="BERT":
        self.feature_dict, msg = self.get_bert_features(
            raw_text=preprocessed_text,
            target_locs=preprocessed_target_locs,
            tokenizer=tokenizer,
            required_features=required_features,
            max_length=max_length,
            label=sentiment,
        )
        # else:
        #     self.feature_dict, msg = self.get_non_bert_features(
        #         raw_text=raw_text,
        #         target_locs=target_locs,
        #         tokenizer=tokenizer,
        #         required_features=required_features,
        #         max_length=max_length,
        #         label=label,
        #     )
        if not self.feature_dict:
            self.succeeded = False
            self.message = f"Features failed: {msg}"

    def pad(self, arrays, max_length, value=0):
        for i in range(len(arrays)):
            d = max_length - len(arrays[i])
            if d >= 0:
                arrays[i] = np.concatenate((arrays[i], [value] * d), axis=None)
            else:
                raise Exception("Array length should not exceed max_length.")
        return arrays

    def get_bert_features(
        self, raw_text, target_locs, tokenizer, required_features, max_length, label=None
    ):

        feature_dict = dict()
        tokens_encoded = tokenizer(
            raw_text,
            max_length=max_length,
            truncation=True,
            padding=True,
            add_special_tokens=True,
        )

        raw_text_ids = np.array(tokens_encoded.input_ids)
        attention_mask = np.array(tokens_encoded.attention_mask)
        token_type_ids = np.array(tokens_encoded.token_type_ids)
        target_mask = np.array([0] * len(raw_text_ids))

        raw_text_ids, attention_mask, token_type_ids, target_mask = self.pad([raw_text_ids, attention_mask, token_type_ids, target_mask], max_length, 0)
        target_pos = []
        for (start_idx, end_idx) in target_locs:
            for char_idx in range(start_idx, end_idx):
                token_idx = tokens_encoded.char_to_token(char_idx)
                if token_idx is not None and token_idx < len(raw_text_ids):
                    target_mask[token_idx] = 1
                    target_pos.append(token_idx)
        # print(target_pos)

        if len(target_pos)==0:
            # print("Target is not found.")
            return None, "Target is not found."
            
        target_pos = np.array(target_pos)

        if "raw_text" in required_features:
            feature_dict["raw_text"] = torch.tensor(raw_text_ids).long()
            feature_dict["tokens"] = tokenizer.convert_ids_to_tokens(feature_dict["raw_text"])

        if "target_mask" in required_features:
            feature_dict["target_mask"] = torch.tensor(target_mask).long()

        if "attention_mask" in required_features:
            feature_dict["attention_mask"] = torch.tensor(attention_mask).long()

        if "token_type_ids" in required_features:
            feature_dict["token_type_ids"] = torch.zeros(len(attention_mask)).long()

        if label is not None:
            label = SENTI_ID_MAP[label]
            feature_dict["label"] = torch.tensor(label).long()

        return feature_dict, ""


class TargetDependentDataset(Dataset):
    def __init__(self, dataset, tokenizer, word_to_idx, args):
        """
        Args:
            data_path (Path object): a list of paths to .json (format of internal label tool)
        """
        self.args = args
        self.model_type = ""
        self.data_config = args.data_config
        self.dataset = dataset
        self.data = []
        self.failed_ids = dict()
        self.tokenizer = tokenizer
        self.word_to_idx = word_to_idx

        self.prepro_config = args.prepro_config
        self.max_length = args.model_config["max_length"]
        MODEL = getattr(sys.modules[__name__], args.train_config["model_class"])
        self.required_features = MODEL.INPUT
        self.load_from_path()

    def load_from_path(self):
        data_path = (
            Path("../data/datasets") / self.data_config["data_dir"] / f"{self.dataset}.json"
        )
        logger.info("***** Loading data *****")
        logger.info("  Data path = %s", str(data_path))
        raw_data = json.load(open(data_path, "r"))
        logger.info("  Number of raw samples = %d", len(raw_data))

        for idx, data_dict in tqdm(enumerate(raw_data)):
            x = TargetDependentExample(
                data_dict=data_dict,
                tokenizer=self.tokenizer,
                prepro_config=self.prepro_config,
                model_type=self.model_type,
                required_features=self.required_features,
                max_length=self.max_length,
            )

            if x.succeeded:
                self.data.append(x)
            else:
                self.failed_ids[idx] = {
                    "data": data_dict,
                    "error": "preprocessing failed"
                }

        logger.info("  Number of loaded samples = %d", len(self.data))

    def get_class_balanced_weights(self):
        class_size = self.df["label"].value_counts()
        weights = 1 / (class_size * class_size.shape[0])
        weights.rename("w", inplace=True)
        df = self.df.join(weights, on="label", how="left")
        return df["w"].tolist(), class_size.max(), class_size.min(), class_size.shape[0]

    def __getitem__(self, index):
        return self.data[index].feature_dict

    def __len__(self):
        return len(self.data)
