# -*- coding: utf-8 -*-
import torch
import numpy as np

from nlp_pipeline.dataset.base import NLPFeature


class SequenceClassificationFeature(NLPFeature):
    def get_feature(
        self, data_dict, tokenizer, required_features, args, diagnosis=False, padding='max_length'
    ):
        diagnosis_dict = dict()

        # data fields
        content = data_dict["content"]
        content2 = data_dict.get("content2", None)
        label = data_dict.get("label", None)

        if "content_da" in data_dict and args.uda_config["use_uda"]:
            use_uda = True # use unsupervised data augmentation
            content_da = data_dict["content_da"]
        else:
            use_uda = False
            content_da = None

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

        if content_da is not None:

            tokens_encoded_da = tokenizer(
                content_da,
                max_length=max_length,
                truncation=True,
                padding=padding,
                add_special_tokens=True,
                return_offsets_mapping=True,
            )

            input_ids_da = tokens_encoded_da.input_ids
            attention_mask_da = tokens_encoded_da.attention_mask
            token_type_ids_da = tokens_encoded_da.token_type_ids
        else:
            input_ids_da = None
            attention_mask_da = None
            token_type_ids_da = None

        if diagnosis:
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            diagnosis_dict["content"] = content
            diagnosis_dict["content2"] = content2
            diagnosis_dict["input_ids"] = input_ids
            diagnosis_dict["tokens"] = tokens
            diagnosis_dict["label"] = label
            diagnosis_dict["label_id"] = label_to_id.get(str(label), None)
            diagnosis_dict["length"] = np.sum(attention_mask)

        if use_uda and input_ids_da is None:
            feature_dict = None

        elif np.sum(attention_mask) == 0:
            feature_dict = None

        else:
            feature_dict = dict()
            if "input_ids" in required_features:
                feature_dict["input_ids"] = torch.tensor(input_ids).long()
                if input_ids_da is not None:
                    feature_dict["input_ids_da"] = torch.tensor(input_ids_da).long()

            if "attention_mask" in required_features:
                feature_dict["attention_mask"] = torch.tensor(attention_mask).long()
                if attention_mask_da is not None:
                    feature_dict["attention_mask_da"] = torch.tensor(attention_mask_da).long()

            if "token_type_ids" in required_features:
                feature_dict["token_type_ids"] = torch.tensor(token_type_ids).long()
                if token_type_ids_da is not None:
                    feature_dict["token_type_ids_da"] = torch.tensor(token_type_ids_da).long()

            if label is not None and label_to_id is not None:
                label = label_to_id[str(label)]
                feature_dict["label"] = torch.tensor(label).long()

        self.feature_dict = feature_dict
        self.diagnosis_dict = diagnosis_dict
        self.tokens_encoded = tokens_encoded
