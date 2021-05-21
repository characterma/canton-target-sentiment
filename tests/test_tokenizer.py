import unittest
import sys
import torch
sys.path.append("../src/")
from tokenizer import InternalTokenizer


class TestTokenizer(unittest.TestCase):
    def test_internal_tokenizer(self):
        raw_text = "hello world 新冠?疫情為全球。帶來"
        word_to_idx = {
            '<OOV>':0,
            '新':1,
            '冠':2, 
            '疫':3,
            '情':4,
            '為':5,
            '全':6,
            'hello': 7,
            'world': 8
        }
        tokenizer = InternalTokenizer(
            word_to_idx=word_to_idx, 
            required_token_types=['CHAR', 'LETTERS']
        )
        encoded = tokenizer(raw_text)

        tokens = ['hello', 'world', '新', '冠', '疫', '情', '為', '全', '球', '帶', '來']
        input_ids = [7,8,1,2,3,4,5,6,0,0,0]
        attention_mask = [1,1,1,1,1,1,1,1,1,1,1]
        self.assertTrue(encoded.tokens==tokens)
        self.assertTrue(encoded.input_ids==input_ids)
        self.assertTrue(encoded.attention_mask==attention_mask)
        self.assertTrue(encoded.char_to_token(13)==3)
        self.assertTrue(tokenizer.convert_ids_to_tokens([1,2])==['新', '冠'])

