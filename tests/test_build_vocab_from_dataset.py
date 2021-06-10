import unittest
import sys
import torch
import json
import os
from pathlib import Path
from collections import namedtuple
sys.path.append("../src/")
from tokenizer import InternalTokenizer, build_vocab_from_dataset


class TestBuildVocabFromDataset(unittest.TestCase):

    args = namedtuple('args', 'data_config prepro_config model_config')
    args.data_config = {'data_dir': '../data/datasets/sample/', "train": "sample_for_build_vocab.json"}
    args.prepro_config = {'steps': []}
    args.model_config = {'vocab_freq_cutoff': 0}
    args.model_dir = Path("./")

    data = [{'content': '標題[心得] Tudor Black Bay 58 M79030B',
            'target_locs': [[7, 12], [13, 18], [19, 22]],
            'sentiment': 'neutral',
            'UNIT_CONTENT': '標題[心得] Tudor Black Bay 58 M79030B',
            'SUBJECT_KEYWORD': 'Tudor,Black,Bay',
            'ASPECT_KEYWORD': '',
            'source_file': 'data1_tw_spam_batch1_rolex_tudor.xlsx',
            'sheent_name': '',
            'annotator': '',
            'sample_id': 21877},
            {'content': '卡地亞，tank很心水，以後一定會買的，不過男表卡地亞有些一言難盡，',
            'target_locs': [[0, 3], [4, 8], [24, 27]],
            'sentiment': 'negative',
            'UNIT_CONTENT': '卡地亞，tank很心水，以後一定會買的，不過男表卡地亞有些一言難盡，',
            'SUBJECT_KEYWORD': '卡地亞,tank,卡地亞',
            'ASPECT_KEYWORD': '',
            'source_file': 'data3_3.xlsx',
            'sheent_name': '',
            'annotator': '',
            'sample_id': 1782}]

    if not os.path.exists("../data/datasets/sample/"):
        os.makedirs("../data/datasets/sample/")
    json.dump(data, open("../data/datasets/sample/sample_for_build_vocab.json", "w"))
    word_to_idx_0 = {'<OOV>': 0,
                    'Bay': 1,
                    '標': 2,
                    '後': 3,
                    '定': 4,
                    '會': 5,
                    '以': 6,
                    'M79030B': 7,
                    '心': 8,
                    '題': 9,
                    '水': 10,
                    '很': 11,
                    '不': 12,
                    '盡': 13,
                    '的': 14,
                    '表': 15,
                    '亞': 16,
                    '買': 17,
                    '一': 18,
                    'tank': 19,
                    '有': 20,
                    '些': 21,
                    'Black': 22,
                    'Tudor': 23,
                    '地': 24,
                    '言': 25,
                    '難': 26,
                    '卡': 27,
                    '男': 28,
                    '過': 29}

    def test_without_frequency_filter(self):
        # case I: no infrequent word filter , check exact value
        self.args.model_config['vocab_freq_cutoff'] = 0
        tokenizer = InternalTokenizer(word_to_idx=None, required_token_types=['CHAR', 'LETTERS'])
        build_vocab_from_dataset(dataset='train', tokenizer=tokenizer,args=self.args)
        self.assertEqual(set(tokenizer.word_to_idx.keys()), set(self.word_to_idx_0.keys()))
        os.system("rm ./word_to_idx.json")

    def test_single_frequency_filter(self):
        # case II: 90% infrequent word filter , check remaining words
        self.args.model_config['vocab_freq_cutoff'] = 0.9 
        tokenizer = InternalTokenizer(word_to_idx=None, required_token_types=['CHAR', 'LETTERS'])
        build_vocab_from_dataset(dataset='train', tokenizer=tokenizer,args=self.args)
        self.assertEqual(len(tokenizer.word_to_idx), 4)
        for w in tokenizer.word_to_idx:
            self.assertIn(w, ['<OOV>', '地', '亞', '卡', '一'])
        os.system("rm ./word_to_idx.json")
        
    def test_increasing_frequency_filter(self):
        # case III: increasing % infrequent word filter , check vocab size
        tokenizer = InternalTokenizer(word_to_idx=None, required_token_types=['CHAR', 'LETTERS'])
        for i in range(1, 10):
            self.args.model_config['vocab_freq_cutoff'] = i / 10
            build_vocab_from_dataset(dataset='train', tokenizer=tokenizer,args=self.args)
            acutal_vocab_size = len(tokenizer.word_to_idx)
            expected_vocab_size = len(self.word_to_idx_0) - int((len(self.word_to_idx_0) - 1) * (i / 10))
            self.assertEqual(acutal_vocab_size, expected_vocab_size)
        os.system("rm ./word_to_idx.json")





