# -*- coding: utf-8 -*-
import torch

from nlp_pipeline.dataset.base import NLPFeature
from nlp_pipeline.dataset.utils import pad_array
from nlp_pipeline.dataset.tagging.utils import get_token_level_tags


class ChineseWordSegmentationFeature(NLPFeature):
    def get_feature(
        self, data_dict, tokenizer, required_features, args, diagnosis=False, padding='max_length'
    ):
        feature_dict = dict()
        diagnosis_dict = dict()

        # data fields
        content = data_dict["content"]
        sent_indexs = data_dict["sent_indexs"]
        postags = data_dict["postags"]

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
            return_length=True,
        )

        input_ids = tokens_encoded.input_ids
        attention_mask = tokens_encoded.attention_mask

        feature_dict["input_ids"] = torch.tensor(input_ids).long()
        feature_dict["attention_mask"] = torch.tensor(attention_mask).long()

        if sent_indexs is not None:
            # tokens_encoded: characters
            label = get_token_level_tags(tokens_encoded, sent_indexs, postags)[
                :max_length
            ]
            # print(label_to_id)
            label_id = [label_to_id[l] for l in label]
            label_id_padded = pad_array(label_id, max_length=max_length, value=0)
            feature_dict["label"] = torch.tensor(label_id_padded).long()

        if diagnosis:
            diagnosis_dict["content"] = content
            diagnosis_dict["input_ids"] = input_ids
            diagnosis_dict["tokens"] = tokenizer.convert_ids_to_tokens(input_ids)
            diagnosis_dict["label"] = label
            diagnosis_dict["label_id"] = label_id

        self.feature_dict = feature_dict
        self.diagnosis_dict = diagnosis_dict
        self.tokens_encoded = tokens_encoded

