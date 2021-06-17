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
from dataset.utils import pad_array


logger = logging.getLogger(__name__)


def get_token_level_tags(tokens_encoded, sent_indexs, postags, scheme='BI'):
    # TODO: sent_indexs -> ent_indexs, postags -> ent_tags
    
    token_tags = dict()
    for tag, (start_idx, end_idx) in zip(postags, sent_indexs):
        for char_idx in range(start_idx, end_idx):
            token_idx = tokens_encoded.char_to_token(char_idx)
            if char_idx==start_idx:
                token_tags[token_idx] = 'B-' + tag          
            else:
                token_tags[token_idx] = 'I-' + tag          

    output = []
    for token_idx in range(len(tokens_encoded)):
        if token_idx in token_tags:
            output.append(token_tags[token_idx])
        else:
            output.append('O')
    return output


def get_word_level_tags(tokens, token_tags):
    words = []
    tags = []

    cur_word = ''
    cur_tag = ''
    for token, tag in zip(tokens, token_tags):
        if tag.startswith('B-'):
            if cur_word:
                words.append(cur_word)
                tags.append(cur_tag)
            cur_word = token 
            cur_tag = tag
        elif tag.startswith('B-') or tag=='O':
            cur_word = cur_word + token 
        else:
            continue

    words.append(cur_word)
    tags.append(cur_tag)
    return words, tags


class ChineseWordSegmentationFeature:
    def __init__(
        self, 
        data_dict, 
        tokenizer, 
        prepro_config, 
        max_length, 
        label_to_id
    ):
        self.succeeded = True
        self.msg = ""
        preprocessor = TextPreprocessor(
            text=data_dict['content'], 
            steps=prepro_config['steps']
        )
        content = preprocessor.preprocessed_text
        words = data_dict.get('words', None)
        postags = data_dict.get('postags', None)
        sent_indexs = data_dict.get('sent_indexs', None)


        self.feature_dict = self.get_features(
            text=content,
            tokenizer=tokenizer,
            max_length=max_length,
            words=words,
            postags=postags,
            sent_indexs=sent_indexs,
            label_to_id=label_to_id,
        )

    @staticmethod
    def get_features(
        text,
        words,
        postags,
        sent_indexs,
        tokenizer,
        max_length,
        label_to_id
        ):
        feature_dict = dict()
        tokens_encoded = tokenizer(text, max_length=max_length, add_special_tokens=False, return_offsets_mapping=False)
        text = tokens_encoded.input_ids
        text = pad_array(text, max_length=max_length, value=0)
        feature_dict['text'] = torch.tensor(text).long()
        attention_mask = tokens_encoded.attention_mask
        attention_mask = pad_array(attention_mask, max_length=max_length, value=0)
        feature_dict['attention_mask'] = torch.tensor(attention_mask).long()
        if sent_indexs is not None:
            # tokens_encoded: characters

            label = get_token_level_tags(tokens_encoded, sent_indexs, postags)
            label = [label_to_id[l] for l in label]
            label = pad_array(label, max_length=max_length, value=0)
            feature_dict['label'] = torch.tensor(label).long()
        return feature_dict


class ChineseWordSegmentationDataset(Dataset):
    def __init__(self, dataset, tokenizer, args):
        self.dataset = dataset
        self.filename = args.data_config[dataset]
        self.args = args
        self.tokenizer = tokenizer
        self.raw_data = [] 
        self.features = []
        self.load_data()

    def __getitem__(self, index):
        return self.features[index]

    def __len__(self):
        return len(self.features)

    def load_data(self):
        data_path = (
            Path(self.args.data_config["data_dir"]) / self.filename
        )
        logger.info("***** Loading data *****")
        logger.info("  Data path = %s", str(data_path))
        self.raw_data = json.load(open(data_path, "r"))
        logger.info("  Number of raw samples = %d", len(self.raw_data))
        for idx, data_dict in tqdm(enumerate(self.raw_data)):

            x = ChineseWordSegmentationFeature(
                data_dict=data_dict,
                tokenizer=self.tokenizer,
                prepro_config=self.args.prepro_config,
                max_length=self.args.model_config["max_length"],
                label_to_id=self.args.label_to_id,
            )

            if x.feature_dict is not None:
                self.features.append(x.feature_dict)

        logger.info("  Number of loaded samples = %d", len(self.features))


    