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


class SequenceClassificationFeature(object):
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
        """
        self.succeeded = True
        self.msg = ""
        self.intermediate = dict()
        raw_text = data_dict['content']
        label = data_dict.get("label", None)
        
        preprocessor = TextPreprocessor(
            text=raw_text, 
            steps=prepro_config['steps']
        )
        preprocessed_text = preprocessor.preprocessed_text
        self.feature_dict, self.diagnosis_dict = self.get_features(
            text=preprocessed_text,
            tokenizer=tokenizer,
            required_features=required_features,
            max_length=max_length,
            label=label,
            label_to_id=label_to_id,
            diagnosis=diagnosis
        )

    @staticmethod
    def get_features(
        text, tokenizer, required_features, max_length, label=None, label_to_id=None, diagnosis=False
    ):
        diagnosis_dict = dict()
        feature_dict = dict()
        tokens_encoded = tokenizer(
            text,
            add_special_tokens=True,
            return_offsets_mapping=True
        )

        input_ids = np.array(tokens_encoded.input_ids)[:max_length]
        attention_mask = np.array(tokens_encoded.attention_mask)[:max_length]
        token_type_ids = np.array(tokens_encoded.token_type_ids)[:max_length]

        input_ids = pad_array(input_ids, max_length, 0)
        attention_mask = pad_array(attention_mask, max_length, 0)
        token_type_ids = pad_array(token_type_ids, max_length, 0)
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        if diagnosis:
            diagnosis_dict['fea_text'] = text
            diagnosis_dict['fea_input_ids'] = input_ids
            diagnosis_dict['fea_tokens'] = tokens

        if "input_ids" in required_features:
            feature_dict["input_ids"] = torch.tensor(input_ids).long()

        if "attention_mask" in required_features:
            feature_dict["attention_mask"] = torch.tensor(attention_mask).long()

        if "token_type_ids" in required_features:
            feature_dict["token_type_ids"] = torch.tensor(token_type_ids).long()

        if label is not None and label_to_id is not None:
            label = label_to_id[label]
            feature_dict["label"] = torch.tensor(label).long()

        return feature_dict, diagnosis_dict


class SequenceClassificationDataset(Dataset):
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
            x = SequenceClassificationFeature(
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
        return statistics

    def insert_predictions(self, predictions):
        self.diagnosis_df['prediction'] = predictions

    def __getitem__(self, index):
        return self.features[index]

    def __len__(self):
        return len(self.features)
