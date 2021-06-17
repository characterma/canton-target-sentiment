import unittest
import sys
import torch
sys.path.append("../src/")
from tokenizer import MultiLingualTokenizer, CharacterSplitTokenizer


class TestTokenizer(unittest.TestCase):
    def test_multilingual_tokenizer(self):
        raw_text = "hello world 新冠?疫情為全球。帶來"
        word_to_id = {
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
        tokenizer = MultiLingualTokenizer(
            word_to_id=word_to_id, 
            required_token_types=['CHAR', 'LETTERS']
        )
        encoded = tokenizer(raw_text)
        tokens = ['hello', 'world', '新', '冠', '疫', '情', '為', '全', '球', '帶', '來']
        input_ids = [7,8,1,2,3,4,5,6,0,0,0]
        attention_mask = [1,1,1,1,1,1,1,1,1,1,1]

        self.assertEqual(encoded.tokens, tokens)
        self.assertEqual(encoded.input_ids, input_ids)
        self.assertEqual(encoded.attention_mask, attention_mask)
        self.assertEqual(encoded.char_to_token(13), 3)
        self.assertEqual(tokenizer.convert_ids_to_tokens([1,2]), ['新', '冠'])

    def test_character_split_tokenizer(self):
            raw_text = "hello world 新冠?疫情為全球。帶來"
            word_to_id = {
                '<OOV>':0,
                '新':1,
                '冠':2, 
                '疫':3,
                '情':4,
                '為':5,
                '全':6,
                'h': 7,
                'e': 8,
                'l': 9,
                'o': 10, 
                'w': 11,
                'r': 12, 
                'd': 13
            }
            tokenizer = CharacterSplitTokenizer(
                word_to_id=word_to_id, 
            )
            encoded = tokenizer(raw_text)
            tokens = ['h', 'e', 'l', 'l', 'o', 'w', 'o', 'r', 'l', 'd', '新', '冠', '?', '疫', '情', '為', '全', '球', '。', '帶', '來']
            input_ids = [7, 8, 9, 9, 10, 11, 10, 12, 9, 13, 1, 2, 0, 3, 4, 5, 6, 0, 0, 0, 0]
            attention_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

            self.assertEqual(encoded.tokens, tokens)
            self.assertEqual(encoded.input_ids, input_ids)
            self.assertEqual(encoded.attention_mask, attention_mask)
            self.assertEqual(encoded.char_to_token(3), 3)
            self.assertEqual(tokenizer.convert_ids_to_tokens([1,2]), ['新', '冠'])
