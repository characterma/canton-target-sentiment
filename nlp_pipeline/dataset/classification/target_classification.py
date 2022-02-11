# -*- coding: utf-8 -*-
import torch
import numpy as np

from nlp_pipeline.dataset.base import NLPFeature


class TargetClassificationFeature(NLPFeature):
    def get_feature(
        self, data_dict, tokenizer, required_features, args, diagnosis=False, padding='max_length'
    ):
        diagnosis_dict = dict()
        feature_dict = dict()

        # data fields
        content = data_dict["content"]
        target_locs = data_dict["target_locs"]
        label = data_dict.get("label", None)

        # params
        max_length = args.model_config["max_length"]
        label_to_id = args.label_to_id
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

        target_mask = np.array([0] * len(input_ids))
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        target_token_loc = []

        for (start_idx, end_idx) in target_locs:
            for char_idx in range(start_idx, end_idx):
                token_idx = tokens_encoded.char_to_token(char_idx)
                target_token_loc.append(token_idx)
                if token_idx is not None and token_idx < max_length:
                    target_mask[token_idx] = 1

        if diagnosis:
            if "raw" in data_dict:
                diagnosis_dict["raw_headline"] = data_dict["raw"].get("headline", None)
                diagnosis_dict["raw_content"] = data_dict["raw"].get("content", None)

            diagnosis_dict["content"] = content
            diagnosis_dict["input_ids"] = input_ids
            diagnosis_dict["target_locs"] = target_locs
            diagnosis_dict["tokens"] = tokens
            diagnosis_dict["target_token_loc"] = target_token_loc
            diagnosis_dict["target_char"] = [
                content[si:ei] for (si, ei) in target_locs
            ]
            diagnosis_dict["target_token"] = []
            diagnosis_dict["label"] = label
            diagnosis_dict["label_id"] = label_to_id.get(str(label), None)
            diagnosis_dict["success"] = True if sum(target_mask) > 0 else False
            diagnosis_dict["error_msg"] = []
            for i in target_token_loc:
                if i is None:
                    diagnosis_dict["target_token"].append("NOT FOUND")
                    diagnosis_dict["error_msg"].append("TARGET NOT FOUND")
                elif i < max_length:
                    diagnosis_dict["target_token"].append(tokens[i])
                else:
                    diagnosis_dict["target_token"].append("> MAX_LEN")
                    diagnosis_dict["error_msg"].append("TARGET > MAX_LEN")

            diagnosis_dict["error_msg"] = sorted(
                list(set(diagnosis_dict["error_msg"]))
            )

        if sum(target_mask) == 0:
            feature_dict = None
        else:

            if "input_ids" in required_features:
                feature_dict["input_ids"] = torch.tensor(input_ids).long()

            if "target_mask" in required_features:
                feature_dict["target_mask"] = torch.tensor(target_mask).long()

            if "attention_mask" in required_features:
                feature_dict["attention_mask"] = torch.tensor(attention_mask).long()

            if "token_type_ids" in required_features:
                feature_dict["token_type_ids"] = torch.tensor(target_mask).long()

            if label is not None and label_to_id is not None:
                label = label_to_id[str(label)]
                feature_dict["label"] = torch.tensor(label).long()

        self.feature_dict = feature_dict
        self.diagnosis_dict = diagnosis_dict
        self.tokens_encoded = tokens_encoded
