#!/usr/bin/python
# -*- coding: utf-8 -*-
import json
import logging
import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from pathlib import Path
from utils import SENTI_ID_MAP, SPEC_TOKEN
from preprocess import preprocess_text_hk_beauty, standardize_text
from sklearn.utils import resample


logger = logging.getLogger(__name__)


def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
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
    """
    A single training/test example for simple sequence classification.
    """

    def __init__(
        self,
        raw_text,
        target_locs,
        tokenizer,
        preprocess_config,
        label=None,
        required_features=[],
        word2idx=None,
        get_vocab_only=False,
        vocab=None,
    ):

        self.raw_text = standardize_text(raw_text)
        self.target_locs = target_locs
        self.succeeded = True

        if preprocess_config.get("text_preprocessing", "") == "hk_beauty":
            self.raw_text, (st_idx, ed_idx), _, _, _, _ = preprocess_text_hk_beauty(
                raw_text, target_locs[0][0], target_locs[0][1], sent_sep=""
            )
            if st_idx is None or ed_idx is None:
                self.succeeded = False
                return None

            self.target_locs = [[st_idx, ed_idx]]

        self.label = label
        self.word2idx = word2idx
        self.tokenizer = tokenizer
        self.preprocess_config = preprocess_config
        self.required_features = required_features
        # self.get_vocab_only = get_vocab_only

        if get_vocab_only:
            self.vocab = TargetDependentExample.get_vocab_non_bert(
                raw_text=self.raw_text, tokenizer=self.tokenizer, vocab=vocab
            )
        else:
            if self.word2idx is None:
                pass
                # self.features = TargetDependentExample.get_features(
                #     tgt_sent=self.tgt_sent,
                #     start_idx=self.start_idx,
                #     end_idx=self.end_idx,
                #     hl_sent=self.hl_sent,
                #     prev_sents=self.prev_sents,
                #     next_sents=self.next_sents,
                #     tgt_in_hl=self.tgt_in_hl,
                #     tokenizer=self.tokenizer,
                #     max_length=self.preprocess_config.get('max_length', 180),
                #     mask_target=self.preprocess_config.get('mask_target', False),
                #     required_features=self.required_features,
                #     label=label
                # )
            else:
                self.features = TargetDependentExample.get_features_non_bert(
                    raw_text=self.raw_text,
                    target_locs=self.target_locs,
                    tokenizer=self.tokenizer,
                    max_length=self.preprocess_config.get("max_length", 180),
                    required_features=self.required_features,
                    label=label,
                    word2idx=word2idx,
                )

    @staticmethod
    def pad(arrays, max_length, value=0):
        for i in range(len(arrays)):
            space = max_length - len(arrays[i])
            assert space >= 0
            if space > 0:
                arrays[i] = np.concatenate((arrays[i], [value] * space), axis=None)
        return arrays

    @staticmethod
    def get_vocab_non_bert(raw_text, tokenizer, vocab):
        if tokenizer is not None:
            tokens = tokenizer(raw_text)
            for t in tokens:
                if t.text in vocab:
                    vocab[t.text] += 1
                else:
                    vocab[t.text] = 1
            return vocab
        else:
            tokens = raw_text.split()
            for t in tokens:
                if t in vocab:
                    vocab[t] += 1
                else:
                    vocab[t] = 1
            return vocab

    @staticmethod
    def get_features_non_bert(
        raw_text,
        target_locs,
        tokenizer,
        max_length,
        required_features,
        word2idx,
        label=None,
    ):
        if tokenizer is not None:

            tokens = tokenizer(raw_text)
            raw_text_ids = [word2idx.get(t.text, word2idx["<OOV>"]) for t in tokens]
            attention_mask = np.zeros(len(raw_text_ids))
            attention_mask[: len(raw_text_ids)] = 1
            target_mask = np.zeros(len(raw_text_ids))

            for (start_idx, end_idx) in target_locs:
                matches = tokens.char_span(start_idx, end_idx, alignment_mode="expand")
                for m in matches:
                    target_mask[m.i] = 1
        else:
            tokens = []
            target_mask = []
            cnt = 0
            for idx, c in enumerate(raw_text):
                if c.strip() != "":
                    tokens.append(c)
                    target_mask.append(0)
                    for (start_idx, end_idx) in target_locs:
                        if start_idx <= idx < end_idx:
                            target_mask[-1] = 1
                            break

            raw_text_ids = [word2idx.get(t, word2idx["<OOV>"]) for t in tokens]
            attention_mask = np.zeros(len(raw_text_ids))
            attention_mask[: len(raw_text_ids)] = 1

        results = {
            "raw_text": torch.tensor(raw_text_ids).long(),
            "attention_mask": torch.tensor(attention_mask).long(),
            "target_mask": torch.tensor(target_mask).long(),
            "tokens": [str(t) for t in tokens],
            "label": SENTI_ID_MAP[label] if label is not None else None,
        }
        return results

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
        label=None,
    ):
        assert len(tgt_sent) > 0
        features = dict()

        tgt_sent_encoded = tokenizer(
            tgt_sent,
            max_length=max_length * 10,
            truncation=True,
            padding=False,
            add_special_tokens=True,
        )

        # Exclude CLS first, append at the end

        raw_text_ids = np.array(tgt_sent_encoded.input_ids[:max_length])
        attention_mask = np.array(tgt_sent_encoded.attention_mask[:max_length])
        token_type_ids = np.array(tgt_sent_encoded.token_type_ids[:max_length])
        target_mask = np.array([0] * len(raw_text_ids))

        # Find the token positions of target
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
                if token_idx is not None and token_idx < len(raw_text_ids):
                    # token_idx = token_idx - 1 # because CLS is removed
                    target_mask[token_idx] = 1
                    tgt_token_ids.append(token_idx)

        tgt_token_ids = np.array(tgt_token_ids)

        cur_len = len(raw_text_ids)
        if max_length - cur_len > 0:
            if tgt_in_hl:
                # if it is in headline, only append next sentences
                if len(next_sents) > 0:
                    next_sents_encoded = tokenizer(
                        next_sents,
                        padding=False,
                        add_special_tokens=True,
                    )
                    raw_text_ids = np.concatenate(
                        (
                            raw_text_ids,
                            next_sents_encoded.input_ids[1:][: max_length - cur_len],
                        ),
                        axis=None,
                    )
                    attention_mask = np.concatenate(
                        (
                            attention_mask,
                            next_sents_encoded.attention_mask[1:][
                                : max_length - cur_len
                            ],
                        ),
                        axis=None,
                    )
                    token_type_ids = np.concatenate(
                        (
                            token_type_ids,
                            next_sents_encoded.token_type_ids[1:][
                                : max_length - cur_len
                            ],
                        ),
                        axis=None,
                    )
                    target_mask = np.concatenate(
                        (
                            target_mask,
                            [0]
                            * len(
                                next_sents_encoded.input_ids[1:][: max_length - cur_len]
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
                # it is not in headline, we need headline, previous sentences, and next sentences.
                if len(hl_sent) > 0:
                    hl_sent_encoded = tokenizer(
                        hl_sent,
                        padding=False,
                        add_special_tokens=True,
                    )

                    hl_sent_len = min(
                        len(hl_sent_encoded.input_ids), max_length - (cur_len - 1)
                    )
                else:
                    hl_sent_encoded = None
                    hl_sent_len = 0  # at least 1, because we need [CLS]

                cur_len += hl_sent_len
                space = max(max_length - cur_len, 0)
                # hl + prev_sents + tgt_sent + next_sents
                # if still has space, try previous sentences and next sentences
                if space > 0 and len(prev_sents) > 0:
                    prev_sents_encoded = tokenizer(
                        prev_sents,
                        padding=False,
                        add_special_tokens=True,
                    )
                    prev_sents_len = (
                        len(prev_sents_encoded.input_ids)
                        if hl_sent_len == 0
                        else len(prev_sents_encoded.input_ids) - 1
                    )
                else:
                    prev_sents_encoded = None
                    prev_sents_len = 0

                if space > 0 and len(next_sents) > 0:
                    next_sents_encoded = tokenizer(
                        next_sents,
                        padding=False,
                        add_special_tokens=True,
                    )
                    next_sents_len = len(next_sents_encoded.input_ids) - 1
                else:
                    next_sents_encoded = None
                    next_sents_len = 0

                # # space excluding headline and target sentence (only previous sentences and next sentences)
                # space = max_length - cur_len - hl_sent_len + 1

                if prev_sents_len < int(space / 2):
                    left_len = prev_sents_len
                    right_len = min(space - prev_sents_len, next_sents_len)
                elif next_sents_len < int((space + 1) / 2):
                    right_len = next_sents_len
                    left_len = min(space - next_sents_len, prev_sents_len)
                else:
                    left_len = min(int(space / 2), prev_sents_len)
                    right_len = min(int((space + 1) / 2), next_sents_len)

                # prev_sents + tgt_sent + next_sents
                if right_len > 0:
                    # something on the right
                    raw_text_ids = np.concatenate(
                        (
                            raw_text_ids,
                            next_sents_encoded.input_ids[1 : right_len + 1],
                        ),  # exclude CLS
                        axis=None,
                    )
                    attention_mask = np.concatenate(
                        (
                            attention_mask,
                            next_sents_encoded.attention_mask[1 : right_len + 1],
                        ),
                        axis=None,
                    )
                    token_type_ids = np.concatenate(
                        (
                            token_type_ids,
                            next_sents_encoded.token_type_ids[1 : right_len + 1],
                        ),
                        axis=None,
                    )
                    target_mask = np.concatenate(
                        (
                            target_mask,
                            [0] * len(next_sents_encoded.input_ids[1 : right_len + 1]),
                        ),
                        axis=None,
                    )

                if left_len > 0:
                    raw_text_ids = np.concatenate(
                        (
                            prev_sents_encoded.input_ids[1:][-left_len:],
                            raw_text_ids[1:],
                        ),  # exclude CLS & include SEP
                        axis=None,
                    )
                    attention_mask = np.concatenate(
                        (
                            prev_sents_encoded.attention_mask[1:][-left_len:],
                            attention_mask[1:],
                        ),
                        axis=None,
                    )
                    token_type_ids = np.concatenate(
                        (
                            prev_sents_encoded.token_type_ids[1:][-left_len:],
                            token_type_ids[1:],
                        ),
                        axis=None,
                    )
                    target_mask = np.concatenate(
                        (
                            [0]
                            * len(prev_sents_encoded.token_type_ids[1:][-left_len:]),
                            target_mask[1:],
                        ),
                        axis=None,
                    )
                    tgt_token_ids = (
                        tgt_token_ids
                        + len(prev_sents_encoded.token_type_ids[1:][-left_len:])
                        - 1
                    )

                if hl_sent_len > 0:
                    raw_text_ids = np.concatenate(
                        (
                            hl_sent_encoded.input_ids[:hl_sent_len],
                            raw_text_ids if left_len > 0 else raw_text_ids[1:],
                        ),
                        axis=None,
                    )
                    attention_mask = np.concatenate(
                        (
                            hl_sent_encoded.attention_mask[:hl_sent_len],
                            attention_mask if left_len > 0 else attention_mask[1:],
                        ),
                        axis=None,
                    )
                    token_type_ids = np.concatenate(
                        (
                            hl_sent_encoded.token_type_ids[:hl_sent_len],
                            token_type_ids if left_len > 0 else token_type_ids[1:],
                        ),
                        axis=None,
                    )
                    target_mask = np.concatenate(
                        (
                            [0] * hl_sent_len,
                            target_mask if left_len > 0 else target_mask[1:],
                        ),
                        axis=None,
                    )
                    tgt_token_ids = (
                        tgt_token_ids + hl_sent_len
                        if left_len > 0
                        else tgt_token_ids + hl_sent_len - 1
                    )

            (
                raw_text_ids,
                attention_mask,
                token_type_ids,
                target_mask,
            ) = TargetDependentExample.pad(
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

        # print(required_features)
        if "raw_text" in required_features:
            features["raw_text"] = torch.tensor(raw_text_ids).long()
            features["tokens"] = tokenizer.convert_ids_to_tokens(features["raw_text"])

        if label_id is not None:
            features["label"] = torch.tensor(label_id).long()

        # if "target" in required_features:
        #     target_ids = raw_text_ids[start_token_pos : end_token_pos + 1]
        #     target_ids = np.concatenate(
        #         (target_ids, [0] * (len(raw_text_ids) - len(target_ids))), axis=None
        #     )
        # features["target"] = torch.tensor(tgt_token_ids).long()
        # features["target_tokens"]= tokenizer.convert_ids_to_tokens(torch.tensor(tgt_token_ids).long())

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
            target_right_inclu = raw_text_ids[attention_mask > 0][start_token_pos:][
                -1::-1
            ]
            target_right_inclu = np.concatenate(
                (
                    target_right_inclu,
                    [0] * (len(raw_text_ids) - len(target_right_inclu)),
                ),
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
            # features["token_type_ids"] = torch.tensor(token_type_ids).long()
            features["token_type_ids"] = torch.zeros(len(attention_mask)).long()
            # features["token_type_ids"][start_token_pos:end_token_pos+1] = 1

        # if "target_span" in required_features:
        features["target_span"] = torch.tensor([start_token_pos, end_token_pos]).long()
        features["target_tokens"] = tokenizer.convert_ids_to_tokens(
            raw_text_ids[start_token_pos : end_token_pos + 1]
        )

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
        required_features=[],
        add_special_tokens=True,
        get_vocab_only=False,
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
        self.timer = timer
        self.required_features = required_features
        self.get_vocab_only = get_vocab_only
        self.vocab = dict()
        self.pad_collate = PadCollate(
            input_cols=required_features,
            pad_cols=["raw_text", "attention_mask", "target_mask"],
            dim=-1,
        )

        if get_vocab_only:
            _ = self.load_from_path(word2idx=word2idx)

        else:
            self.data = self.load_from_path(word2idx=word2idx)
            self.df = pd.DataFrame(
                data={
                    "raw_text": [e.raw_text for e in self.data],
                    "tokens": [e.features["tokens"] for e in self.data],
                    "label": [e.label for e in self.data],
                }
            )

        # count_class = {}
        # for e in self.data:
        #     label = e.label
        #     if label not in count_class:
        #         count_class[label] = 1
        #     else:
        #         count_class[label] += 1
        # for k, v in count_class.items():
        #     logger.info(f"  Number of '{k}' = {v}")

    def load_from_path(self, word2idx=None):
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

                # print(self.label_map[label])
                e = TargetDependentExample(
                    raw_text=doc["content"],
                    target_locs=[(t["start_ind"], t["end_ind"])],
                    tokenizer=self.tokenizer,
                    label=self.label_map[label],
                    preprocess_config=self.preprocess_config,
                    required_features=self.required_features,
                    word2idx=word2idx,
                    get_vocab_only=self.get_vocab_only,
                    vocab=self.vocab,
                )

                if e.succeeded:
                    data.append(e)

        #         if e.preprocess_succeeded:
        #             if e.feature_succeeded:
        #                 data.append(e)
        #             else:
        #                 feature_failed += 1
        #         else:
        #             preprocess_failed += 1

        # # resample
        # if self.preprocess_config.get("resample_size", None):
        #     data = resample(data, replace=True, n_samples=int(self.preprocess_config["resample_size"]))

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
        return self.data[index].features

    def __len__(self):
        return len(self.data)
