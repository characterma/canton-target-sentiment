#!/usr/bin/python
# -*- coding: utf-8 -*-
import copy
import csv
import json
import logging
import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, Dataset
from pathlib import Path
from cantonsa.utils import get_label_map
from cantonsa.constants import SENTI_ID_MAP, SENTI_ID_MAP_INV, SPEC_TOKEN
from cantonsa.preprocess import preprocess_text_hk_beauty, get_mask_target
from transformers import AutoTokenizer
from sklearn.utils import resample


logger = logging.getLogger(__name__)


class TargetDependentExample(object):
    """
    A single training/test example for simple sequence classification.
    """

    def __init__(
        self,
        raw_text,
        raw_start_idx,
        raw_end_idx,
        tokenizer, 
        preprocess_config,
        label=None,
        required_features=[]
    ):

        self.raw_text = raw_text
        self.raw_start_idx = raw_start_idx
        self.raw_end_idx = raw_end_idx
        self.label = label
        self.tokenizer = tokenizer
        self.preprocess_config = preprocess_config
        self.required_features = required_features

        # text preprocessing
        self.tgt_sent, self.start_idx, self.end_idx, self.hl_sent, self.prev_sents, self.next_sents, self.tgt_in_hl = TargetDependentExample.preprocess_text(
            raw_text=self.raw_text, 
            raw_start_idx=self.raw_start_idx, 
            raw_end_idx=self.raw_end_idx, 
            text_preprocessing=self.preprocess_config.get('text_preprocessing', None), 
            mask_target=self.preprocess_config.get('mask_target', False), 
        )

        if self.start_idx is None:
            self.preprocess_succeeded = False
        else:
            self.preprocess_succeeded = True
        
            # feature engineering
            self.features = TargetDependentExample.get_features(
                tgt_sent=self.tgt_sent, 
                start_idx=self.start_idx, 
                end_idx=self.end_idx, 
                hl_sent=self.hl_sent, 
                prev_sents=self.prev_sents, 
                next_sents=self.next_sents, 
                tgt_in_hl=self.tgt_in_hl, 
                tokenizer=self.tokenizer,
                max_length=self.preprocess_config.get('max_length', 180), 
                mask_target=self.preprocess_config.get('mask_target', False), 
                required_features=self.required_features
            )

            if len(self.features) > 0:
                self.feature_succeeded = True
            else:
                self.feature_succeeded = False

    @staticmethod
    def preprocess_text(raw_text, raw_start_idx, raw_end_idx, text_preprocessing="", mask_target=False):

        if text_preprocessing == "hk_beauty":
            (
                tgt_sent,
                (start_idx, end_idx),
                hl_sent,
                prev_sents,
                next_sents,
                tgt_in_hl,
            ) = preprocess_text_hk_beauty(
                raw_text,
                raw_start_idx,
                raw_end_idx,
            )
        else:
            tgt_sent, start_idx, end_idx = (
                raw_text,
                raw_start_idx,
                raw_end_idx,
            )
            hl_sent = ""
            prev_sents = ""
            next_sents = ""
            tgt_in_hl = False


        if mask_target:
            tgt_sent, (start_idx, end_idx) = get_mask_target(
                tgt_sent, start_idx, end_idx
            )

        return tgt_sent, start_idx, end_idx, hl_sent, prev_sents, next_sents, tgt_in_hl

    @staticmethod
    def pad(arrays, max_length):
        for i in range(len(arrays)):
            space = max_length - len(arrays[i])
            assert space >= 0
            if space > 0:
                arrays[i] = np.concatenate((arrays[i], [0] * space), axis=None)
        return arrays

    @staticmethod
    def get_features(
        tgt_sent, 
        start_idx, 
        end_idx, 
        hl_sent, 
        prev_sents, 
        next_sents, 
        tgt_in_hl, 
        tokenizer,
        max_length, 
        mask_target, 
        required_features, 
        label=None
    ):
        features = {}

        tgt_sent_encoded = tokenizer(
            tgt_sent,
            max_length=max_length * 10,
            truncation=True,
            padding=False,
            add_special_tokens=True,
        )

        raw_text_ids = np.array(tgt_sent_encoded.input_ids)
        attention_mask = np.array(tgt_sent_encoded.attention_mask)
        token_type_ids = np.array(tgt_sent_encoded.token_type_ids)
        target_mask = np.array([0] * len(raw_text_ids))

        tgt_token_ids = []
        if mask_target:
            mask_id = tokenizer.convert_tokens_to_ids(SPEC_TOKEN.TARGET)
            for pos, token_idx in enumerate(raw_text_ids):
                if token_idx == mask_id:
                    target_mask[pos] = 1
                    tgt_token_ids.append(pos)
        else:
            for char_idx in range(start_idx, end_idx):
                token_idx = tgt_sent_encoded.char_to_token(char_idx)
                if token_idx is not None:
                    target_mask[token_idx] = 1
                    tgt_token_ids.append(token_idx)

        tgt_token_ids = np.array(tgt_token_ids)

        cur_len = len(raw_text_ids)
        if max_length - len(raw_text_ids) > 0:
            if tgt_in_hl:
                next_sents_encoded = tokenizer(
                    tgt_sent,
                    padding=False,
                    add_special_tokens=True,
                )
                if len(next_sents_encoded.input_ids) > 0:
                    raw_text_ids = np.concatenate(
                        (
                            raw_text_ids,
                            next_sents_encoded.input_ids[1:][-(max_length - cur_len) :],
                        ),
                        axis=None,
                    )
                    attention_mask = np.concatenate(
                        (
                            attention_mask,
                            next_sents_encoded.attention_mask[1:][
                                -(max_length - cur_len) :
                            ],
                        ),
                        axis=None,
                    )
                    token_type_ids = np.concatenate(
                        (
                            token_type_ids,
                            next_sents_encoded.token_type_ids[1:][
                                -(max_length - cur_len) :
                            ],
                        ),
                        axis=None,
                    )
                    target_mask = np.concatenate(
                        (
                            target_mask,
                            [0]
                            * len(
                                next_sents_encoded.input_ids[1:][
                                    -(max_length - cur_len) :
                                ]
                            ),
                        ),
                        axis=None,
                    )
                    assert (
                        len(raw_text_ids)
                        == len(attention_mask)
                        == len(token_type_ids)
                        == len(target_mask)
                    )
            else:
                tgt_token_ids = tgt_token_ids - 1 # Because [CLS] in tgt_sent will be removed
                hl_sent_encoded = tokenizer(
                    hl_sent,
                    padding=False,
                    add_special_tokens=True,
                )

                hl_sent_len = len(hl_sent_encoded.input_ids)
                if max_length - cur_len + 1 > hl_sent_len:
                    # hl + prev_sents + tgt_sent + next_sents
                    prev_sents_encoded = tokenizer(
                        prev_sents,
                        padding=False,
                        add_special_tokens=True,
                    )
                    next_sents_encoded = tokenizer(
                        next_sents,
                        padding=False,
                        add_special_tokens=True,
                    )

                    prev_sents_len = len(prev_sents_encoded.input_ids) - 1
                    next_sents_len = len(next_sents_encoded.input_ids) - 1
                    space = max_length - cur_len - hl_sent_len + 1

                    if prev_sents_len < int(space / 2):
                        left = int(space / 2)
                        right = space - prev_sents_len
                    elif next_sents_len < int((space + 1) / 2):
                        right = int((space + 1) / 2)
                        left = space - next_sents_len
                    else:
                        left = int(space / 2)
                        right = int((space + 1) / 2)
                    # print(left + right , space, cur_len, hl_sent_len)

                    # prev_sents + tgt_sent + next_sents
                    raw_text_ids = np.concatenate(
                        (raw_text_ids[1:], next_sents_encoded.input_ids[1 : right + 1]),
                        axis=None,
                    )
                    attention_mask = np.concatenate(
                        (
                            attention_mask[1:],
                            next_sents_encoded.attention_mask[1 : right + 1],
                        ),
                        axis=None,
                    )
                    token_type_ids = np.concatenate(
                        (
                            token_type_ids[1:],
                            next_sents_encoded.token_type_ids[1 : right + 1],
                        ),
                        axis=None,
                    )
                    target_mask = np.concatenate(
                        (
                            target_mask[1:],
                            [0] * len(next_sents_encoded.input_ids[1 : right + 1]),
                        ),
                        axis=None,
                    )

                    # print("--", len(raw_text_ids))

                    if left > 0:
                        raw_text_ids = np.concatenate(
                            (prev_sents_encoded.input_ids[1:][-left:], raw_text_ids),
                            axis=None,
                        )
                        attention_mask = np.concatenate(
                            (
                                prev_sents_encoded.attention_mask[1:][-left:],
                                attention_mask,
                            ),
                            axis=None,
                        )
                        token_type_ids = np.concatenate(
                            (
                                prev_sents_encoded.token_type_ids[1:][-left:],
                                token_type_ids,
                            ),
                            axis=None,
                        )
                        target_mask = np.concatenate(
                            (
                                [0]
                                * len(prev_sents_encoded.token_type_ids[1:][-left:]),
                                target_mask,
                            ),
                            axis=None,
                        )
                        # print("--", len(raw_text_ids))
                        tgt_token_ids = tgt_token_ids + len(
                            prev_sents_encoded.token_type_ids[1:][-left:]
                        )

                    # hl + prev_sents + tgt_sent + next_sents
                    raw_text_ids = np.concatenate(
                        (hl_sent_encoded.input_ids, raw_text_ids), axis=None
                    )
                    attention_mask = np.concatenate(
                        (hl_sent_encoded.attention_mask, attention_mask), axis=None
                    )
                    token_type_ids = np.concatenate(
                        (hl_sent_encoded.token_type_ids, token_type_ids), axis=None
                    )
                    target_mask = np.concatenate(
                        ([0] * hl_sent_len, target_mask), axis=None
                    )
                    tgt_token_ids = tgt_token_ids + hl_sent_len
                    # print("--", len(raw_text_ids))

                else:
                    # hl + tgt_sent
                    space = max_length - cur_len
                    raw_text_ids = np.concatenate(
                        (hl_sent_encoded.input_ids[: space + 1], raw_text_ids[1:]),
                        axis=None,
                    )
                    attention_mask = np.concatenate(
                        (
                            hl_sent_encoded.attention_mask[: space + 1],
                            attention_mask[1:],
                        ),
                        axis=None,
                    )
                    token_type_ids = np.concatenate(
                        (
                            hl_sent_encoded.token_type_ids[: space + 1],
                            token_type_ids[1:],
                        ),
                        axis=None,
                    )
                    target_mask = np.concatenate(
                        (
                            [0] * len(hl_sent_encoded.token_type_ids[: space + 1]),
                            target_mask[1:],
                        ),
                        axis=None,
                    )
                    tgt_token_ids = tgt_token_ids + len(
                        hl_sent_encoded.token_type_ids[: space + 1]
                    )

            raw_text_ids, attention_mask, token_type_ids, target_mask = TargetDependentExample.pad(
                arrays=[raw_text_ids, attention_mask, token_type_ids, target_mask],
                max_length=max_length,
            )

            assert (
                len(raw_text_ids)
                == len(attention_mask)
                == len(token_type_ids)
                == len(target_mask)
                == max_length
            )
        else:
            raw_text_ids = raw_text_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            token_type_ids = token_type_ids[:max_length]
            target_mask = target_mask[:max_length]
            assert (
                len(raw_text_ids)
                == len(attention_mask)
                == len(token_type_ids)
                == len(target_mask)
                == max_length
            )

        if label is not None:
            label_id = SENTI_ID_MAP[label]
        else:
            label_id = None

        # print(tgt_token_ids)
        start_token_pos = min(tgt_token_ids) if len(tgt_token_ids) > 0 else None
        end_token_pos = max(tgt_token_ids) if len(tgt_token_ids) > 0 else None

        if start_token_pos is None or end_token_pos is None:
            return features

        if "raw_text_without_target" in required_features:
            raw_text_without_target = np.concatenate(
                (raw_text_ids[:start_token_pos], raw_text_ids[end_token_pos + 1 :]),
                axis=None,
            )
            raw_text_without_target = np.concatenate(
                (
                    raw_text_without_target,
                    [0] * (len(raw_text_ids) - len(raw_text_without_target)),
                ),
                axis=None,
            )
            features["raw_text_without_target"] = torch.tensor(
                raw_text_without_target
            ).long()

        if "raw_text" in required_features:
            features["raw_text"] = torch.tensor(raw_text_ids).long()

        if label_id is not None:
            features["label"] = torch.tensor(label_id).long()

        if "target" in required_features:
            target_ids = raw_text_ids[start_token_pos : end_token_pos + 1]
            target_ids = np.concatenate(
                (target_ids, [0] * (len(raw_text_ids) - len(target_ids))), axis=None
            )
            features["target"] = torch.tensor(target_ids).long()

        if "target_right" in required_features:  
            target_right = raw_text_ids[attention_mask > 0][end_token_pos:][-1::-1]
            target_right = np.concatenate(
                (target_right, [0] * (len(raw_text_ids) - len(target_right))), axis=None
            )
            features["target_right"] = torch.tensor(target_right).long()

        if "target_left" in required_features:  
            target_left = raw_text_ids[attention_mask > 0][:start_token_pos]
            target_left = np.concatenate(
                (target_left, [0] * (len(raw_text_ids) - len(target_left))), axis=None
            )
            features["target_left"] = torch.tensor(target_left).long()

        if "target_right_inclu" in required_features:  
            target_right_inclu = raw_text_ids[attention_mask > 0][start_token_pos:][-1::-1]
            target_right_inclu = np.concatenate(
                (target_right_inclu, [0] * (len(raw_text_ids) - len(target_right_inclu))),
                axis=None,
            )
            features["target_right_inclu"] = torch.tensor(target_right_inclu).long()

        if "target_left_inclu" in required_features: 
            target_left_inclu = raw_text_ids[attention_mask > 0][: end_token_pos + 1]
            target_left_inclu = np.concatenate(
                (target_left_inclu, [0] * (len(raw_text_ids) - len(target_left_inclu))),
                axis=None,
            )        
            features["target_left_inclu"] = torch.tensor(target_left_inclu).long()

        if "target_mask" in required_features:     
            features["target_mask"] = torch.tensor(target_mask).long()

        if "attention_mask" in required_features:     
            features["attention_mask"] = torch.tensor(attention_mask).long()

        if "token_type_ids" in required_features:     
            features["token_type_ids"] = torch.tensor(token_type_ids).long()

        if "target_span" in required_features:     
            features["target_span"] = torch.tensor([start_token_pos, end_token_pos]).long()

        # self.tokens = tokenizer.convert_ids_to_tokens(raw_text_ids)
        # self.tgt_tokens = tokenizer.convert_ids_to_tokens(raw_text_ids[start_token_pos : end_token_pos + 1])

        return features


class TargetDependentDataset(Dataset):
    def __init__(
        self, 
        data_path=None,
        label_map=None,
        tokenizer=None,
        preprocess_config=None,
        word2idx=None,
        timer=None, 
        name="",
        required_features=[]
    ):
        """
        Args:
            data_path (Path object): a list of paths to .json (format of internal label tool)
        """
        self.name = name
        self.word2idx = word2idx
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.preprocess_config = preprocess_config
        self.max_length = preprocess_config["max_length"]
        self.text_preprocessing = preprocess_config.get("text_preprocessing", None)
        self.mask_target = preprocess_config.get("mask_target", False)
        self.label_map = label_map
        self._add_new_word = False
        self.timer = timer
        self.required_features = required_features

        if word2idx is not None and len(self.word2idx) == 0:
            self.word2idx[None] = 0
            self.word2idx["<OOV>"] = 1
            self._add_new_word = True

        self.data = self.load_from_path()

        self.df = pd.DataFrame(
            data={
                "raw_text": [e.raw_text for e in self.data],
                "tgt_sent": [e.tgt_sent for e in self.data],
                "hl_sent": [e.hl_sent for e in self.data],
                "prev_sents": [e.prev_sents for e in self.data],
                "next_sents": [e.next_sents for e in self.data],
                "tgt_in_hl": [e.tgt_in_hl for e in self.data],
                "label": [e.label for e in self.data],
            }
        )

        count_class = {}
        for e in self.data:
            label = e.label
            if label not in count_class:
                count_class[label] = 1
            else:
                count_class[label] += 1
        for k, v in count_class.items():
            logger.info(f"  Number of '{k}' = {v}")

    def load_from_path(self):
        logger.info("***** Loading data *****")
        logger.info("  Path = %s", str(self.data_path))
        data = []
        key = 0
        lines = open(self.data_path, "rb").readlines()
        logger.info("  Number of lines = %d", len(lines))

        preprocess_failed = 0
        feature_failed = 0

        data = []

        for line in tqdm(lines):
            doc = json.loads(line)
            for t in doc["labels"]:
                label = str(t["label_name"])

                if label not in self.label_map and label.lower() not in SENTI_ID_MAP:
                    logger.warning("  Illegal label : %s", label)
                    continue

                e = TargetDependentExample(
                    raw_text=doc['content'],
                    raw_start_idx=t['start_ind'],
                    raw_end_idx=t['end_ind'],
                    tokenizer=self.tokenizer, 
                    label=self.label_map[label],
                    preprocess_config=self.preprocess_config,
                    required_features=self.required_features                   
                )

                if e.preprocess_succeeded:
                    if e.feature_succeeded:
                        data.append(e)
                    else:
                        feature_failed += 1
                else:
                    preprocess_failed += 1

        # resample
        if self.preprocess_config.get("resample_size", None):
            data = resample(data, replace=True, n_samples=int(self.preprocess_config["resample_size"]))

        logger.info("  Loaded examples = %d", len(data))
        logger.info("  Failed preprocessing = %d", preprocess_failed)
        logger.info("  Failed max len = %d", feature_failed)

        return data

    def get_class_balanced_weights(self):
        class_size = self.df["label"].value_counts()
        weights = 1 / (class_size * class_size.shape[0])
        weights.rename("w", inplace=True)
        df = self.df.join(weights, on="label", how="left")
        return df["w"].tolist(), class_size.max(), class_size.min(), class_size.shape[0]

    def __getitem__(self, index):
        return self.data[index].data

    def __len__(self):
        return len(self.data)
