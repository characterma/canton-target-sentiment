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
from cantonsa.preprocess import preprocess_text_hk_beauty, mask_target
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)


class TDSAExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        label: sentiment type of the target.
        start_idx: starting char of the target
        end_idx: ending char of the target
    """

    def __init__(
        self,
        raw_text,
        tgt_sent,
        start_idx,
        end_idx,
        hl_sent="",
        prev_sents="",
        next_sents="",
        tgt_in_hl=False,
        label="unknown",
        docid="_",
        key=0,
    ):

        self.raw_text = raw_text
        self.tgt_sent = tgt_sent
        self.hl_sent = hl_sent
        self.prev_sents = prev_sents
        self.next_sents = next_sents
        self.tgt_in_hl = tgt_in_hl

        self.label = label
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.target = tgt_sent[start_idx:end_idx]
        self.docid = docid
        self.is_feature = False

        # LM features
        self.data = {"key": key}

    @staticmethod
    def pad(arrays, max_length):
        for i in range(len(arrays)):
            space = max_length - len(arrays[i])
            assert space >= 0
            if space > 0:
                arrays[i] = np.concatenate((arrays[i], [0] * space), axis=None)
        return arrays

    def convert_to_features(
        self,
        tokenizer,
        max_length,
        add_special_tokens=True,
        word2idx=None,
        add_new_word=False,
        mask_target=False
    ):
        tgt_sent_encoded = tokenizer(
            self.tgt_sent,
            max_length=max_length * 10,
            truncation=True,
            padding=False,
            add_special_tokens=add_special_tokens,
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
            for char_idx in range(self.start_idx, self.end_idx):
                token_idx = tgt_sent_encoded.char_to_token(char_idx)
                if token_idx is not None:
                    target_mask[token_idx] = 1
                    tgt_token_ids.append(token_idx)

        if len(tgt_token_ids) == 0:
            self.is_feature = False
            return self.is_feature

        tgt_token_ids = np.array(tgt_token_ids)

        cur_len = len(raw_text_ids)
        if max_length - len(raw_text_ids) > 0:
            if self.tgt_in_hl:
                next_sents_encoded = tokenizer(
                    self.tgt_sent,
                    padding=False,
                    add_special_tokens=add_special_tokens,
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
                    self.hl_sent,
                    padding=False,
                    add_special_tokens=add_special_tokens,
                )

                hl_sent_len = len(hl_sent_encoded.input_ids)
                if max_length - cur_len + 1 > hl_sent_len:
                    # hl + prev_sents + tgt_sent + next_sents
                    prev_sents_encoded = tokenizer(
                        self.prev_sents,
                        padding=False,
                        add_special_tokens=add_special_tokens,
                    )
                    next_sents_encoded = tokenizer(
                        self.next_sents,
                        padding=False,
                        add_special_tokens=add_special_tokens,
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

            raw_text_ids, attention_mask, token_type_ids, target_mask = self.pad(
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

        # token_type_ids[-1] = 1 is OK for BERT
        # token_type_ids[-1] = 2 is NOT OK for BERT

        # if word2idx is not None:
        #     if not add_new_word:
        #         raw_text_ids = np.array([word2idx.get(t, word2idx["<OOV>"]) for t in encoded.tokens()])
        #     else:
        #         raw_text_ids = []
        #         for t in encoded.tokens():
        #             if t not in word2idx:
        #                 word2idx[t] = int(len(word2idx))
        #             raw_text_ids.append(word2idx[t])
        #         raw_text_ids = np.array(raw_text_ids)

        label_id = SENTI_ID_MAP[self.label]

        # print(tgt_token_ids)
        start_token_pos = min(tgt_token_ids) if len(tgt_token_ids) > 0 else None
        end_token_pos = max(tgt_token_ids) if len(tgt_token_ids) > 0 else None

        if start_token_pos is None or end_token_pos is None:
            self.is_feature = False
            return self.is_feature

        target_ids = raw_text_ids[start_token_pos : end_token_pos + 1]
        target_left = raw_text_ids[attention_mask > 0][:start_token_pos]
        target_right = raw_text_ids[attention_mask > 0][end_token_pos:][-1::-1]
        target_left_inclu = raw_text_ids[attention_mask > 0][: end_token_pos + 1]
        target_right_inclu = raw_text_ids[attention_mask > 0][start_token_pos:][-1::-1]
        raw_text_without_target = np.concatenate(
            (raw_text_ids[:start_token_pos], raw_text_ids[end_token_pos + 1 :]),
            axis=None,
        )

        target_left = np.concatenate(
            (target_left, [0] * (len(raw_text_ids) - len(target_left))), axis=None
        )
        target_right = np.concatenate(
            (target_right, [0] * (len(raw_text_ids) - len(target_right))), axis=None
        )
        target_ids = np.concatenate(
            (target_ids, [0] * (len(raw_text_ids) - len(target_ids))), axis=None
        )
        target_left_inclu = np.concatenate(
            (target_left_inclu, [0] * (len(raw_text_ids) - len(target_left_inclu))),
            axis=None,
        )
        target_right_inclu = np.concatenate(
            (target_right_inclu, [0] * (len(raw_text_ids) - len(target_right_inclu))),
            axis=None,
        )
        raw_text_without_target = np.concatenate(
            (
                raw_text_without_target,
                [0] * (len(raw_text_ids) - len(raw_text_without_target)),
            ),
            axis=None,
        )

        self.data["raw_text"] = torch.tensor(raw_text_ids).long()
        self.data["raw_text_without_target"] = torch.tensor(
            raw_text_without_target
        ).long()
        self.data["label"] = torch.tensor(label_id).long()
        self.data["target"] = torch.tensor(target_ids).long()
        self.data["target_right"] = torch.tensor(target_right).long()
        self.data["target_left"] = torch.tensor(target_left).long()
        self.data["target_right_inclu"] = torch.tensor(target_right_inclu).long()
        self.data["target_left_inclu"] = torch.tensor(target_left_inclu).long()
        self.data["target_mask"] = torch.tensor(target_mask).long()
        self.data["attention_mask"] = torch.tensor(attention_mask).long()
        self.data["token_type_ids"] = torch.tensor(token_type_ids).long()
        self.data["target_span"] = torch.tensor([start_token_pos, end_token_pos]).long()

        # self.data["key"] = key
        self.tokens = tokenizer.convert_ids_to_tokens(raw_text_ids)
        self.tgt_tokens = tokenizer.convert_ids_to_tokens(raw_text_ids[start_token_pos : end_token_pos + 1])

        self.is_feature = True
        return self.is_feature

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class TDSADataset(Dataset):
    def __init__(
        self,
        data_path,
        label_map,
        tokenizer,
        preprocess_config,
        word2idx=None,
        add_special_tokens=True,
        to_df=True,  # set False to accelerate
        show_statistics=True,  # set False to accelerate
        name="",
    ):
        """
        Args:
            data_path (Path object): a list of paths to .json (format of internal label tool)
        """
        self.name = name
        self.word2idx = word2idx
        self.add_special_tokens = add_special_tokens
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = preprocess_config["max_length"]
        self.text_preprocessing = preprocess_config.get("text_preprocessing", None)
        self.mask_target = preprocess_config.get("mask_target", False)
        self.label_map = label_map
        self._add_new_word = False

        if word2idx is not None and len(self.word2idx) == 0:
            self.word2idx[None] = 0
            self.word2idx["<OOV>"] = 1
            self._add_new_word = True

        self.data = self.load_data()

        if to_df:
            self.df = self.to_df()
        else:
            self.df = None

        if show_statistics:
            count_class = {}
            for e in self.data:
                label = e.label
                if label not in count_class:
                    count_class[label] = 1
                else:
                    count_class[label] += 1
            for k, v in count_class.items():
                logger.info(f"  Number of '{k}' = {v}")

    def load_data(self):
        logger.info("***** Loading data *****")
        logger.info("  Path = %s", str(self.data_path))
        data = []
        key = 0
        lines = open(self.data_path, "rb").readlines()
        logger.info("  Number of lines = %d", len(lines))
        failed_prep = 0
        failed_max_len = 0
        for line in tqdm(lines):
            doc = json.loads(line)
            for t in doc["labels"]:
                label = str(t["label_name"])

                if label not in self.label_map and label.lower() not in SENTI_ID_MAP:
                    logger.warning("  Illegal label : %s", label)
                    continue

                tgt_in_hl = True
                prev_sents = ""
                next_sents = ""
                hl_sent = ""

                if not self.text_preprocessing:
                    tgt_sent, start_idx, end_idx = (
                        doc["content"],
                        t["start_ind"],
                        t["end_ind"],
                    )
                elif self.text_preprocessing == "hk_beauty":
                    (
                        tgt_sent,
                        (start_idx, end_idx),
                        hl_sent,
                        prev_sents,
                        next_sents,
                        tgt_in_hl,
                    ) = preprocess_text_hk_beauty(
                        doc["content"],
                        tgt_st_idx=t["start_ind"],
                        tgt_ed_idx=t["end_ind"],
                    )
                else:
                    raise (Exception)

                if start_idx is None or end_idx is None:
                    failed_prep += 1
                    continue

                if self.mask_target:
                    tgt_sent, (start_idx, end_idx) = mask_target(
                        tgt_sent, start_idx, end_idx
                    )

                example = TDSAExample(
                    raw_text=doc["content"],
                    hl_sent=hl_sent,
                    tgt_sent=tgt_sent,
                    prev_sents=prev_sents,
                    next_sents=next_sents,
                    tgt_in_hl=tgt_in_hl,
                    label=self.label_map[label]
                    if label in self.label_map
                    else label.lower(),
                    start_idx=start_idx,
                    end_idx=end_idx,
                    docid=doc["docid"],
                    key=key,
                )

                if example.convert_to_features(
                    self.tokenizer,
                    self.max_length,
                    word2idx=self.word2idx,
                    add_special_tokens=self.add_special_tokens,
                    add_new_word=self._add_new_word,
                    mask_target=self.mask_target
                ):
                    data.append(example)
                else:
                    failed_max_len += 1
                key += 1

        logger.info("  Loaded data = %d", len(data))
        logger.info("  Failed preprocessing = %d", failed_prep)
        logger.info("  Failed max len = %d", failed_max_len)
        return data

    def get_df(self):
        return self.df

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

    def to_df(self):
        df = pd.DataFrame(
            data={
                "key": [e.data["key"] for e in self.data],
                "docid": [e.docid for e in self.data],
                "raw_text": [e.raw_text for e in self.data],
                "tgt_sent": [e.tgt_sent for e in self.data],
                "hl_sent": [e.hl_sent for e in self.data],
                "prev_sents": [e.prev_sents for e in self.data],
                "next_sents": [e.next_sents for e in self.data],
                "tgt_in_hl": [e.tgt_in_hl for e in self.data],
                "tokens": [e.tokens for e in self.data],
                "target": [e.target for e in self.data],
                "tgt_tokens": [e.tgt_tokens for e in self.data],
                "label": [e.label for e in self.data],
            }
        )
        return df
