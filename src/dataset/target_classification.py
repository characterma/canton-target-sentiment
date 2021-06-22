# -*- coding: utf-8 -*-
import json
import logging
import random
import torch
import importlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from pathlib import Path
from preprocess import TextPreprocessor
from collections import Counter
from model import *
from dataset.utils import pad_array, get_model_inputs


logger = logging.getLogger(__name__)


class TargetClassificationFeature(object):
    def __init__(
        self,
        data_dict,
        tokenizer,
        prepro_config,
        required_features,
        max_length,
        label_to_id=None,
        word_to_id=None,
        diagnosis=False
    ):
        """
        data_dict: dict
        prepro_config: dict
        required_features: List
        """
        self.succeeded = True
        self.msg = ""
        self.intermediate = dict()
        text = data_dict['content']
        target_locs = data_dict["target_locs"]
        sentiment = data_dict.get("sentiment", None)
        
        preprocessor = TextPreprocessor(
            text=text, 
            target_locs=target_locs, 
            steps=prepro_config['steps']
        )
        preprocessed_text = preprocessor.preprocessed_text
        preprocessed_target_locs = preprocessor.preprocessed_target_locs
        self.feature_dict, self.diagnosis_dict = self.get_features(
            text=preprocessed_text,
            target_char_loc=preprocessed_target_locs,
            tokenizer=tokenizer,
            required_features=required_features,
            max_length=max_length,
            label=sentiment,
            label_to_id=label_to_id,
            diagnosis=diagnosis
        )

    @staticmethod
    def get_features(
        text, target_char_loc, tokenizer, required_features, max_length, label=None, label_to_id=None, diagnosis=False
    ):
        diagnosis_dict = dict()
        feature_dict = dict()
        tokens_encoded = tokenizer(
            text,
            # max_length=max_length,
            # truncation=True,
            add_special_tokens=True,
            return_offsets_mapping=True
        )

        input_ids = np.array(tokens_encoded.input_ids)[:max_length]
        attention_mask = np.array(tokens_encoded.attention_mask)[:max_length]
        token_type_ids = np.array(tokens_encoded.token_type_ids)[:max_length]
        target_mask = np.array([0] * len(input_ids))
        input_ids = pad_array(input_ids, max_length, 0)
        attention_mask = pad_array(attention_mask, max_length, 0)
        token_type_ids = pad_array(token_type_ids, max_length, 0)
        target_mask = pad_array(target_mask, max_length, 0)
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        target_token_loc = []

        for (start_idx, end_idx) in target_char_loc:
            for char_idx in range(start_idx, end_idx):
                token_idx = tokens_encoded.char_to_token(char_idx)
                target_token_loc.append(token_idx)
                if token_idx is not None and token_idx < max_length:
                    target_mask[token_idx] = 1

        if diagnosis:
            diagnosis_dict['fea_text'] = text
            diagnosis_dict['fea_input_ids'] = input_ids
            diagnosis_dict['fea_target_char_loc'] = target_char_loc
            diagnosis_dict['fea_tokens'] = tokens
            diagnosis_dict['fea_target_token_loc'] = target_token_loc
            diagnosis_dict['fea_target_char'] = [text[si:ei] for (si, ei) in target_char_loc]
            diagnosis_dict['fea_target_token'] = []
            diagnosis_dict['fea_success'] = True if sum(target_mask)>0 else False
            diagnosis_dict['fea_error_msg'] = []
            for i in target_token_loc:
                if i is None:
                    diagnosis_dict['fea_target_token'].append("NOT FOUND")
                    diagnosis_dict['fea_error_msg'].append("TARGET NOT FOUND")
                elif i < max_length:
                    diagnosis_dict['fea_target_token'].append(tokens[i])
                else:
                    diagnosis_dict['fea_target_token'].append("> MAX_LEN")
                    diagnosis_dict['fea_error_msg'].append("TARGET > MAX_LEN")

            diagnosis_dict['fea_error_msg'] = sorted(list(set(diagnosis_dict['fea_error_msg'])))

        if sum(target_mask)==0:
            return None, diagnosis_dict

        if "input_ids" in required_features:
            feature_dict["input_ids"] = torch.tensor(input_ids).long()

        if "target_mask" in required_features:
            feature_dict["target_mask"] = torch.tensor(target_mask).long()

        if "attention_mask" in required_features:
            feature_dict["attention_mask"] = torch.tensor(attention_mask).long()

        if "token_type_ids" in required_features:
            feature_dict["token_type_ids"] = torch.tensor(target_mask).long()

        if label is not None and label_to_id is not None:
            label = label_to_id[label]
            feature_dict["label"] = torch.tensor(label).long()

        return feature_dict, diagnosis_dict


class TargetClassificationDataset(Dataset):
    def __init__(self, dataset, tokenizer, args):
        """
        """
        self.args = args
        self.dataset = dataset
        self.filename = args.data_config[dataset]
        self.raw_data = None
        self.features = []
        self.diagnosis = []
        self.tokenizer = tokenizer
        self.required_features = get_model_inputs(args)
        self.load_data()
        self.diagnosis_df = pd.DataFrame(data=self.diagnosis)

    def load_data(self):
        data_path = (
            self.args.data_dir / self.filename
        )
        logger.info("***** Loading data *****")
        logger.info("  Data path = %s", str(data_path))
        self.raw_data = json.load(open(data_path, "r"))
        logger.info("  Number of raw samples = %d", len(self.raw_data))

        for idx, data_dict in tqdm(enumerate(self.raw_data)):
            diagnosis_dict = dict(zip(["raw_" + k for k in data_dict.keys()], data_dict.values()))
            x = TargetClassificationFeature(
                data_dict=data_dict,
                tokenizer=self.tokenizer,
                prepro_config=self.args.prepro_config,
                required_features=self.required_features,
                max_length=self.args.model_config["max_length"],
                label_to_id=self.args.label_to_id,
                diagnosis=True
            )
            if x.feature_dict is not None:
                self.features.append(x.feature_dict)
            diagnosis_dict.update(x.diagnosis_dict)
            self.diagnosis.append(diagnosis_dict)
        logger.info("  Number of loaded samples = %d", len(self.features))

    def get_data_analysis(self):
        statistics = {'dataset': self.dataset}
        statistics['total_samples'] = self.diagnosis_df.shape[0]
        statistics['fea_success'] = self.diagnosis_df[self.diagnosis_df['fea_success']].shape[0]
        for s in self.diagnosis_df['raw_sentiment'].unique():
            df = self.diagnosis_df[self.diagnosis_df['raw_sentiment']==s]
            statistics[f"raw_{s}"] = df.shape[0]
            statistics[f"fea_{s}"] = df[df['fea_success']].shape[0]
        return statistics

    def insert_predictions(self, predictions):
        pass

    def __getitem__(self, index):
        return self.features[index]

    def __len__(self):
        return len(self.features)
