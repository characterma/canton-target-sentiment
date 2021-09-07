

# -*- coding: utf-8 -*-
import torch
import numpy as np
from dataset.base import NLPFeature


class WinogradSchemaChallengeFeature(NLPFeature):
    def __init__(self, data_dict, tokenizer, args, diagnosis=False):
        super().__init__(
            data_dict=data_dict, tokenizer=tokenizer, args=args, diagnosis=diagnosis
        )

    def get_feature(
        self, data_dict, tokenizer, required_features, args, diagnosis=False
    ):
        diagnosis_dict = dict()
        feature_dict = dict()

        # data fields
        content = data_dict["content"]
        target_locs1 = data_dict["target_locs1"]
        target_locs2 = data_dict["target_locs2"]

        label = data_dict.get("label", None)

        # params
        max_length = args.model_config["max_length"]
        label_to_id = args.label_to_id

        tokens_encoded = tokenizer(
            content,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            add_special_tokens=True,
            return_offsets_mapping=True,
        )

        input_ids = tokens_encoded.input_ids
        attention_mask = tokens_encoded.attention_mask
        token_type_ids = tokens_encoded.token_type_ids

        # target 1
        target_mask1 = np.array([0] * len(input_ids))
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        target_token_loc1 = []

        for (start_idx, end_idx) in target_locs1:
            for char_idx in range(start_idx, end_idx):
                token_idx = tokens_encoded.char_to_token(char_idx)
                target_token_loc1.append(token_idx)
                if token_idx is not None and token_idx < max_length:
                    target_mask1[token_idx] = 1

        # target 2
        target_mask2 = np.array([0] * len(input_ids))
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        target_token_loc2 = []

        for (start_idx, end_idx) in target_locs2:
            for char_idx in range(start_idx, end_idx):
                token_idx = tokens_encoded.char_to_token(char_idx)
                target_token_loc2.append(token_idx)
                if token_idx is not None and token_idx < max_length:
                    target_mask2[token_idx] = 1

        if diagnosis:
            # tokens1 = tokenizer.convert_ids_to_tokens(input_ids1)
            diagnosis_dict["content"] = content
            diagnosis_dict["input_ids"] = input_ids
            diagnosis_dict["tokens"] = tokens
            diagnosis_dict["label"] = label

        if np.sum(attention_mask) == 0:
            return None, diagnosis_dict
        else:

            feature_dict["input_ids"] = torch.tensor(input_ids).long()
            feature_dict["attention_mask"] = torch.tensor(attention_mask).long()
            feature_dict["token_type_ids"] = torch.tensor(token_type_ids).long()

            feature_dict["target_mask1"] = torch.tensor(target_mask1).long()
            feature_dict["target_mask2"] = torch.tensor(target_mask2).long()

            if label is not None and label_to_id is not None:
                label = label_to_id[str(label)]
                feature_dict["label"] = torch.tensor(label).long()
            return feature_dict, diagnosis_dict
