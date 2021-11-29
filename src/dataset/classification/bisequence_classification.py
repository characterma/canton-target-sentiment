# -*- coding: utf-8 -*-
import torch
import numpy as np
from dataset.base import NLPFeature


class BiSequenceClassificationFeature(NLPFeature):
    def get_feature(
        self, data_dict, tokenizer, required_features, args, diagnosis=False, padding='max_length'
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

        if diagnosis:
            diagnosis_dict["content1"] = content1
            diagnosis_dict["content2"] = content2
            diagnosis_dict["label"] = label
            diagnosis_dict["label_id"] = label_to_id.get(str(label), None)

        # Single sequence model
        if ("input_ids" in required_features
            and "attention_mask" in required_features
            and "token_type_ids" in token_type_ids):

            content = content1 + "[SEP]" + content2
            tokens_encoded = tokenizer(
                content,
                max_length=max_length,
                truncation=True,
                padding=padding,
                add_special_tokens=True,
                return_offsets_mapping=True,
            )

            input_ids = tokens_encoded.input_ids
            attention_mask = tokens_encoded.attention_mask
            token_type_ids = tokens_encoded.token_type_ids

            if np.sum(attention_mask) == 0:
                return None, diagnosis_dict

            feature_dict["input_ids"] = torch.tensor(input_ids).long()
            feature_dict["attention_mask"] = torch.tensor(attention_mask).long()
            feature_dict["token_type_ids"] = torch.tensor(token_type_ids).long()
        else:
            pass 

        if label is not None and label_to_id is not None:
            label = label_to_id[str(label)]
            feature_dict["label"] = torch.tensor(label).long()
        return feature_dict, diagnosis_dict
