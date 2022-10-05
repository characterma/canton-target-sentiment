# -*- coding: utf-8 -*-
import torch

from nlp_pipeline.dataset.base import NLPFeature


class SequenceLabelingFeature(NLPFeature):
    def get_feature(
        self, data_dict, tokenizer, required_features, args, diagnosis=False, padding='max_length'
    ):
        feature_dict = dict()
        diagnosis_dict = dict()

        # data fields
        content = data_dict["content"]
        tokens = data_dict["tokens"]
        labels = data_dict["labels"]

        # params
        max_length = args.model_config["max_length"]
        label_to_id = args.label_to_id

        tokens = tokens[:max_length-2]
        input_ids = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(tokens) + [tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)

        if labels is not None:
            labels = ["O"] + labels[:max_length-2] + ["O"]
            label_ids = [label_to_id[l] for l in labels]

        padding_length = max_length - len(input_ids)
        if padding_length > 0:
            input_ids += [tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length

            if labels is not None:
                label_ids += [label_to_id["O"]] * padding_length

        feature_dict["input_ids"] = torch.tensor(input_ids).long()
        feature_dict["attention_mask"] = torch.tensor(attention_mask).long()

        if labels is not None:
            feature_dict["label"] = torch.tensor(label_ids).long()

        if diagnosis:
            diagnosis_dict["content"] = content
            diagnosis_dict["input_ids"] = input_ids
            diagnosis_dict["tokens"] = tokenizer.convert_ids_to_tokens(input_ids)
            diagnosis_dict["label"] = labels
            diagnosis_dict["label_id"] = label_ids

        self.feature_dict = feature_dict
        self.diagnosis_dict = diagnosis_dict
