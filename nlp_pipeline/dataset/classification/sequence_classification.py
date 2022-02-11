# -*- coding: utf-8 -*-
import torch
import numpy as np

from nlp_pipeline.dataset.base import NLPFeature


class SequenceClassificationFeature(NLPFeature):
    def get_feature(
        self, data_dict, tokenizer, required_features, args, diagnosis=False, padding='max_length'
    ):
        diagnosis_dict = dict()
        feature_dict = dict()

        # data fields
        content = data_dict["content"]
        content2 = data_dict.get("content2", None)
        label = data_dict.get("label", None)

        # params
        max_length = args.model_config["max_length"]
        label_to_id = args.label_to_id

        tokens_encoded = tokenizer(
            content,
            content2, 
            max_length=max_length,
            truncation=True,
            padding=padding,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )

        input_ids = tokens_encoded.input_ids
        attention_mask = tokens_encoded.attention_mask
        token_type_ids = tokens_encoded.token_type_ids

        if diagnosis:
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            diagnosis_dict["content"] = content
            diagnosis_dict["content2"] = content2
            diagnosis_dict["input_ids"] = input_ids
            diagnosis_dict["tokens"] = tokens
            diagnosis_dict["label"] = label
            diagnosis_dict["label_id"] = label_to_id.get(str(label), None)
            diagnosis_dict["length"] = np.sum(attention_mask)

        if np.sum(attention_mask) == 0:
            return None, diagnosis_dict
        else:
            if "input_ids" in required_features:
                feature_dict["input_ids"] = torch.tensor(input_ids).long()

            if "attention_mask" in required_features:
                feature_dict["attention_mask"] = torch.tensor(attention_mask).long()

            if "token_type_ids" in required_features:
                feature_dict["token_type_ids"] = torch.tensor(token_type_ids).long()

            if label is not None and label_to_id is not None:
                label = label_to_id[str(label)]
                feature_dict["label"] = torch.tensor(label).long()

        self.feature_dict = feature_dict
        self.diagnosis_dict = diagnosis_dict
        self.tokens_encoded = tokens_encoded
