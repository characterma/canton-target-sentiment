import unittest
import sys
import torch

from nlp_pipeline.tokenizer import MultiLingualTokenizer, CharacterSplitTokenizer
# passed on 2022-02-16

class TestTokenizer(unittest.TestCase):
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
