import unittest
import sys
import torch
sys.path.append("../src/")
from dataset.tagging.chinese_word_segmentation import get_token_level_tags
from transformers import AutoTokenizer


class TestGetTokenLevelTags(unittest.TestCase):
    def test_get_token_level_tags(self):
        
        text = "天氣很好, Thank you!"
        sent_indexs = [[0,2], [2,3], [3,4], [4,5], [5,6], [6,11], [11,12], [12,15], [15,16]]
        postags = ['N','N','N','P','P','E', 'P', 'E','P']
        correct_tags = ['O', 'B-N', 'I-N', 'B-N', 'B-N', 'B-P', 'B-E', 'B-E', 'B-P', 'O']
        tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-chinese", 
            use_fast=True, 
        )

        tokens_encoded = tokenizer(text, return_length=True, add_special_tokens=True)
        tags = get_token_level_tags(
            tokens_encoded=tokens_encoded, 
            sent_indexs=sent_indexs, 
            postags=postags, 
            scheme='BI'
        )
        self.assertEqual(tags, correct_tags)
