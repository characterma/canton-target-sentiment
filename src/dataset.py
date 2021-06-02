# -*- coding: utf-8 -*-
import json
import logging
import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from pathlib import Path
from preprocess import TextPreprocessor
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
        required_features,
        max_length,
        label_to_id=None,
        word_to_idx=None,
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
        raw_text = data_dict['content']
        target_locs = data_dict["target_locs"]
        sentiment = data_dict.get("sentiment", None)
        
        preprocessor = TextPreprocessor(
            text=raw_text, 
            target_locs=target_locs, 
            steps=prepro_config['steps']
        )
        preprocessed_text = preprocessor.preprocessed_text
        preprocessed_target_locs = preprocessor.preprocessed_target_locs
        self.feature_dict, self.diagnosis_dict = self.get_features(
            raw_text=preprocessed_text,
            target_char_loc=preprocessed_target_locs,
            tokenizer=tokenizer,
            required_features=required_features,
            max_length=max_length,
            label=sentiment,
            label_to_id=label_to_id,
            diagnosis=diagnosis
        )

    @staticmethod
    def pad(arrays, max_length, value=0):
        for i in range(len(arrays)):
            d = max_length - len(arrays[i])
            if d >= 0:
                arrays[i] = np.concatenate((arrays[i], [value] * d), axis=None)
            else:
                raise Exception("Array length should not exceed max_length.")
        return arrays

    @staticmethod
    def get_features(
        raw_text, target_char_loc, tokenizer, required_features, max_length, label=None, label_to_id=None, diagnosis=False
    ):
        diagnosis_dict = dict()
        feature_dict = dict()
        tokens_encoded = tokenizer(
            raw_text,
            # max_length=max_length,
            # truncation=True,
            add_special_tokens=True,
            return_offsets_mapping=True
        )

        raw_text_ids = np.array(tokens_encoded.input_ids)[:max_length]
        attention_mask = np.array(tokens_encoded.attention_mask)[:max_length]
        token_type_ids = np.array(tokens_encoded.token_type_ids)[:max_length]
        target_mask = np.array([0] * len(raw_text_ids))
        raw_text_ids, attention_mask, token_type_ids, target_mask = TargetDependentExample.pad([raw_text_ids, attention_mask, token_type_ids, target_mask], max_length, 0)
        tokens = tokenizer.convert_ids_to_tokens(raw_text_ids)
        target_token_loc = []

        for (start_idx, end_idx) in target_char_loc:
            for char_idx in range(start_idx, end_idx):
                token_idx = tokens_encoded.char_to_token(char_idx)
                target_token_loc.append(token_idx)
                if token_idx is not None and token_idx < max_length:
                    target_mask[token_idx] = 1

        if diagnosis:
            diagnosis_dict['fea_text'] = raw_text
            diagnosis_dict['fea_text_ids'] = raw_text_ids
            diagnosis_dict['fea_target_char_loc'] = target_char_loc
            diagnosis_dict['fea_tokens'] = tokens
            diagnosis_dict['fea_target_token_loc'] = target_token_loc
            diagnosis_dict['fea_target_char'] = [raw_text[si:ei] for (si, ei) in target_char_loc]
            diagnosis_dict['fea_target_token'] = []
            diagnosis_dict['fea_success'] = True
            diagnosis_dict['fea_error_msg'] = []
            for i, (st_char_idx, ed_char_idx) in zip():
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
                diagnosis_dict['fea_success'] = False


        if sum(target_mask)==0:
            return None, diagnosis_dict

        if "raw_text" in required_features:
            feature_dict["raw_text"] = torch.tensor(raw_text_ids).long()

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


def load_vocab(tokenizer, vocab_path, args):
    logger.info("***** Loading vocab *****")
    word_to_idx = json.load(open(vocab_path, 'r'))
    tokenizer.update_word_idx(word_to_idx)
    args.vocab_size = len(word_to_idx)
    logger.info("  Vocab size = %d", len(word_to_idx))


def build_vocab_from_pretrained(tokenizer, args):
    emb_path = Path("../data/word_embeddings") / args.model_config['pretrained_word_emb']
    words = []
    logger.info("***** Building vocab from pretrained *****")
    logger.info("  Embedding path = %s", str(emb_path))
    with open(emb_path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            break
        for line in tqdm(f):
            words.append(line.split(' ')[0])
    word_to_idx = {}
    word_to_idx['<OOV>'] = 0
    word_to_idx.update(dict(zip(words, range(1, len(words) + 1))))
    logger.info("  Vocab size = %d", len(word_to_idx))
    json.dump(word_to_idx, open(args.model_dir / 'word_to_idx.json', 'w'))
    tokenizer.update_word_idx(word_to_idx)
    args.vocab_size = len(word_to_idx)


def build_vocab_from_dataset(dataset, tokenizer, args):
    data_path = (
        Path(args.data_config["data_dir"]) / f"{dataset}.json"
    )
    raw_data = json.load(open(data_path, "r"))
    logger.info("***** Building vocab from dataset *****")
    logger.info("  Data path = %s", str(data_path))
    logger.info("  Number of raw samples = %d", len(raw_data))
    all_words = []
    word_to_idx = dict()

    for idx, data_dict in tqdm(enumerate(raw_data)):
        preprocessor = TextPreprocessor(
            text=data_dict['content'], 
            target_locs=data_dict['target_locs'], 
            steps=args.prepro_config['steps']
        )
        preprocessed_text = preprocessor.preprocessed_text
        all_words.extend(tokenizer(raw_text=preprocessed_text, max_length=None).tokens)
    word_counter = Counter(all_words)

    vocab_freq_cutoff = args.model_config['vocab_freq_cutoff']
    words = list(set(all_words))
    random.shuffle(words)
    words = sorted(words, key=lambda w: word_counter[w])
    infreq_words = words[:int(vocab_freq_cutoff * len(words))]
    logger.info("  Number of infrequency words = %d", len(infreq_words))
    json.dump(infreq_words, open("words.json", 'w'))

    word_to_idx['<OOV>'] = 0
    cur_idx = 1
    for w in words:
        if w not in infreq_words:
            word_to_idx[w] = cur_idx
            cur_idx += 1

    logger.info("  Infrequenct words = %d", len(infreq_words))
    logger.info("  Vocab size = %d", len(word_to_idx))

    json.dump(word_to_idx, open(args.model_dir / 'word_to_idx.json', 'w'))
    tokenizer.update_word_idx(word_to_idx)
    args.vocab_size = len(word_to_idx)


class TargetDependentDataset(Dataset):
    def __init__(self, dataset, tokenizer, args):
        """
        Args:
            data_path (Path object): a list of paths to .json (format of internal label tool)
        """
        self.args = args
        self.dataset = dataset
        self.raw_data = None
        self.features = []
        self.diagnosis = []
        self.tokenizer = tokenizer
        Model = getattr(sys.modules[__name__], args.train_config["model_class"])
        self.required_features = Model.INPUT
        self.load_from_path()
        self.diagnosis_df = pd.DataFrame(data=self.diagnosis)

    def load_from_path(self):
        data_path = (
            Path(self.args.data_config["data_dir"]) / f"{self.dataset}.json"
        )
        logger.info("***** Loading data *****")
        logger.info("  Data path = %s", str(data_path))
        self.raw_data = json.load(open(data_path, "r"))
        logger.info("  Number of raw samples = %d", len(self.raw_data))

        for idx, data_dict in tqdm(enumerate(self.raw_data)):
            diagnosis_dict = dict(zip(["raw_" + k for k in data_dict.keys()], data_dict.values()))
            x = TargetDependentExample(
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

    def get_class_balanced_weights(self):
        class_size = self.df["label"].value_counts()
        weights = 1 / (class_size * class_size.shape[0])
        weights.rename("w", inplace=True)
        df = self.df.join(weights, on="label", how="left")
        return df["w"].tolist(), class_size.max(), class_size.min(), class_size.shape[0]

    def __getitem__(self, index):
        return self.features[index]

    def __len__(self):
        return len(self.features)
