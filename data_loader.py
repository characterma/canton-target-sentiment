import copy
import csv
import json
import logging
import os
import pandas as pd
import torch
from torch.utils.data import TensorDataset, Dataset
from pathlib import Path
from utils import get_label_map, SENTI_ID_MAP
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)


class SingleTargetExample(object):
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

    def __init__(self, text, start_idx, end_idx, label="unknown", docid="_", key=0):
        self.text = text
        self.label = label
        self.start_idx = start_idx 
        self.end_idx = end_idx
        self.entity = text[start_idx:end_idx]
        self.docid = docid
        self.key = key

        # LM features
        self.input_ids = None
        self.attention_mask = None
        self.token_type_ids = None
        self.label_id = None
        self.t_mask = None
        self.is_feature = False

    def converted_to_features(self, tokenizer, max_length):
        encoded = tokenizer(self.text, 
                            max_length=max_length, 
                            padding="max_length", 
                            truncation=True)
        input_ids = encoded.input_ids
        token_type_ids = encoded.token_type_ids if 'token_type_ids' in encoded else []
        attention_mask = encoded.attention_mask

        label_id = SENTI_ID_MAP[self.label]
        target_tokens = []
        t_mask = [0] * len(attention_mask)

        if not self.start_idx > max_length:

            for idx in range(self.start_idx, self.end_idx):
                token_pos = encoded.char_to_token(idx)
                if token_pos is not None:
                    t_mask[token_pos] = 1
            # print(len(input_ids))

            self.input_ids = input_ids
            self.attention_mask = attention_mask
            self.token_type_ids = token_type_ids
            self.label_id = label_id
            self.t_mask = t_mask
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


def load_examples(data_path, label_map, to_feature=False, tokenizer=None, max_length=None):
    examples = []
    # label_map = get_label_map(label_map_path)
    key = 0
    for line in open(data_path, "rb").readlines():
        doc = json.loads(line)
        for t in doc['labels']:
            label = t['label_name']
            if label not in label_map:
                continue

            example = SingleTargetExample(
                    text = doc['content'], 
                    label = label_map[label], 
                    start_idx = t['start_ind'], 
                    end_idx = t['end_ind'], 
                    docid = doc['docid'], 
                    key = key, 
            )
            if to_feature and example.converted_to_features(tokenizer, max_length):
                examples.append(example)
            key += 1   
    return examples 


def load_dataset(data_path, label_map_path, tokenizer, max_length, details=False):
    """
    Args:
        data_path (Path object): a list of paths to .json (format of internal label tool)
    Output:
        examples: a list of SingleTargetExample instances.
    """
    examples = load_examples(data_path, label_map_path, to_feature=True, tokenizer=tokenizer, max_length=max_length)
    all_keys = torch.tensor([f.key for f in examples], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in examples], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in examples], dtype=torch.long
    )
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in examples], dtype=torch.long
    )
    all_label_ids = torch.tensor([f.label_id for f in examples], dtype=torch.long)
    all_t_mask = torch.tensor(
        [f.t_mask for f in examples], dtype=torch.long
    )
    dataset = TensorDataset(
        all_keys, 
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        all_label_ids,
        all_t_mask,
    )
    return dataset, examples_to_table(examples) if details else None


def examples_to_table(examples):

    df = pd.DataFrame(data={
        "key": [e.key for e in examples], 
        "text": [e.text for e in examples], 
        "entity": [e.entity for e in examples], 
        "start_idx": [e.start_idx for e in examples], 
        "end_idx": [e.end_idx for e in examples], 
        "label": [e.label for e in examples], 
    })

    return df



# if __name__=="__main__":
#     tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased", use_fast=True, padding_side ='right')
#     data_path = Path(os.environ['DATA_PATH']) / "label_tool_data" / "train.json"
#     features = load_dataset(tokenizer, 'valid', **{'max_length': 120})
#     print(len(features), "features.")
    