# -*- coding: utf-8 -*-
import torch
import numpy as np
from dataset.base import NLPFeature


class BiSequenceClassificationFeature(NLPFeature):
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
        content1 = data_dict["content1"]
        content2 = data_dict["content2"]
        label = data_dict.get("label", None)

        # params
        max_length = args.model_config["max_length"]
        label_to_id = args.label_to_id

        tokens_encoded1 = tokenizer(
            content1,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            add_special_tokens=True,
            return_offsets_mapping=True,
        )

        tokens_encoded2 = tokenizer(
            content2,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            add_special_tokens=True,
            return_offsets_mapping=True,
        )

        input_ids1 = tokens_encoded1.input_ids
        input_ids2 = tokens_encoded2.input_ids

        attention_mask1 = tokens_encoded1.attention_mask
        attention_mask2 = tokens_encoded2.attention_mask

        token_type_ids1 = tokens_encoded1.token_type_ids
        token_type_ids2 = tokens_encoded2.token_type_ids

        if diagnosis:
            tokens1 = tokenizer.convert_ids_to_tokens(input_ids1)
            tokens2 = tokenizer.convert_ids_to_tokens(input_ids2)
            diagnosis_dict["content1"] = content1
            diagnosis_dict["content2"] = content2

            diagnosis_dict["input_ids1"] = input_ids1
            diagnosis_dict["input_ids2"] = input_ids2

            diagnosis_dict["tokens1"] = tokens1
            diagnosis_dict["tokens2"] = tokens2

            diagnosis_dict["label"] = label

        if np.sum(attention_mask1) == 0 or np.sum(attention_mask2) == 0:
            return None, diagnosis_dict
        else:

            feature_dict["input_ids1"] = torch.tensor(input_ids1).long()
            feature_dict["input_ids2"] = torch.tensor(input_ids2).long()

            feature_dict["attention_mask1"] = torch.tensor(attention_mask1).long()
            feature_dict["attention_mask2"] = torch.tensor(attention_mask2).long()

            feature_dict["token_type_ids1"] = torch.tensor(token_type_ids1).long()
            feature_dict["token_type_ids2"] = torch.tensor(token_type_ids2).long()

            if label is not None and label_to_id is not None:
                label = label_to_id[str(label)]
                feature_dict["label"] = torch.tensor(label).long()
            return feature_dict, diagnosis_dict
