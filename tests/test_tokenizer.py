import unittest
import sys
import torch
import json
import os 

from collections import namedtuple 
from pathlib import Path, PurePath

from nlp_pipeline.tokenizer import get_tokenizer, MultiLingualTokenizer, CharacterSplitTokenizer
from nlp_pipeline.utils import load_config


class TestTokenizer(unittest.TestCase):
    test_dir = Path(PurePath(__file__).parent).resolve()

    @classmethod
    def tearDownClass(cls):
        os.system(f"rm -rf {cls.test_dir}/test_end_to_end_samples/9/result")
        os.system(f"rm -rf {cls.test_dir}/test_end_to_end_samples/9/model")
        os.system(f"rm -rf {cls.test_dir}/test_end_to_end_samples/9/logs")
        os.system(f"rm {cls.test_dir}/test_end_to_end_samples/9/log")

    def test_extra_special_tokens(self):
        args = namedtuple('args', 'config_dir')
        args.config_dir = Path(f"{self.test_dir}/test_end_to_end_samples/9")
        args = load_config(args)
        tokenizer = get_tokenizer(args=args)
        special_tokens_map = json.load(
            open(args.config_dir / "model" / "tokenizer" / "special_tokens_map.json", 'r')
        )
        for sp_tkn in ["[unused5]", "[unused4]", "[unused3]", "[unused2]", "[unused1]"]:
            self.assertTrue(sp_tkn in special_tokens_map["additional_special_tokens"])

    def test_multilingual_tokenizer(self):
        raw_text = "hello world 新冠?疫情為全球。帶來"
        word_to_id = {
            '<OOV>':0,
            '<PAD>':1,
            '新':2,
            '冠':3, 
            '疫':4,
            '情':5,
            '為':6,
            '全':7,
            'hello': 8,
            'world': 9
        }
        tokenizer = MultiLingualTokenizer(
            word_to_id=word_to_id, 
            required_token_types=['CHAR', 'LETTERS']
        )
        encoded = tokenizer(raw_text)
        tokens = ['hello', 'world', '新', '冠', '疫', '情', '為', '全', '球', '帶', '來']
        input_ids = [8, 9, 2, 3, 4, 5, 6, 7, 1, 1, 1]
        attention_mask = [1,1,1,1,1,1,1,1,1,1,1]

        self.assertEqual(encoded.tokens, tokens)
        self.assertEqual(encoded.input_ids, input_ids)
        self.assertEqual(encoded.attention_mask, attention_mask)
        self.assertEqual(encoded.char_to_token(13), 3)
        self.assertEqual(tokenizer.convert_ids_to_tokens([2, 3]), ['新', '冠'])

    def test_character_split_tokenizer(self):
            raw_text = "hello world 新冠?疫情為全球。帶來"
            word_to_id = {
                '<OOV>':0,
                '<PAD>':1,
                '新':2,
                '冠':3, 
                '疫':4,
                '情':5,
                '為':6,
                '全':7,
                'h': 8,
                'e': 9,
                'l': 10,
                'o': 11, 
                'w': 12,
                'r': 13, 
                'd': 14
            }
            tokenizer = CharacterSplitTokenizer(
                word_to_id=word_to_id, 
            )
            encoded = tokenizer(raw_text)
            tokens = ['h', 'e', 'l', 'l', 'o', 'w', 'o', 'r', 'l', 'd', '新', '冠', '?', '疫', '情', '為', '全', '球', '。', '帶', '來']
            input_ids = [8, 9, 10, 10, 11, 12, 11, 13, 10, 14, 2, 3, 1, 4, 5, 6, 7, 1, 1, 1, 1]
            attention_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

            self.assertEqual(encoded.tokens, tokens)
            self.assertEqual(encoded.input_ids, input_ids)
            self.assertEqual(encoded.attention_mask, attention_mask)
            self.assertEqual(encoded.char_to_token(3), 3)
            self.assertEqual(tokenizer.convert_ids_to_tokens([2, 3]), ['新', '冠'])
