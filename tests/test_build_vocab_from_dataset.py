import unittest
import sys
import torch
import json
import os
from pathlib import Path, PurePath
from collections import namedtuple

from nlp_pipeline.tokenizer import MultiLingualTokenizer, build_vocab_from_dataset


class TestBuildVocabFromDataset(unittest.TestCase):

    test_dir = Path(PurePath(__file__).parent).resolve()
    src_dir = test_dir.parent / "nlp_pipeline"
    config_dir = test_dir.parent / "config"

    args = namedtuple('args', 'data_config prepro_config model_config')
    args.data_config = {
        'data_dir': '../data/datasets/sample/', 
        "train": "sample_for_build_vocab1.json", 
        "dev": "sample_for_build_vocab2.json", 
    }
    args.data_dir = src_dir / args.data_config['data_dir']
    args.prepro_config = {'steps': []}
    args.model_config = {'vocab_freq_cutoff': 0}
    args.model_dir = Path("./")

    data1 = [{'content': '標題[心得] Tudor Black Bay 58 M79030B'},
            {'content': '卡地亞，tank很心水，以後一定會買的，不過男表卡地亞有些一言難盡，'}]

    data2 = [{'content': '竞赛'}]

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    json.dump(data1, open(args.data_dir / "sample_for_build_vocab1.json", "w"))
    json.dump(data2, open(args.data_dir / "sample_for_build_vocab2.json", "w"))
    word_to_id_0 = {'<OOV>': 0,
                    '<PAD>': 1,
                    'Bay': 2,
                    '標': 3,
                    '後': 4,
                    '定': 5,
                    '會': 6,
                    '以': 7,
                    'M79030B': 8,
                    '心': 9,
                    '題': 10,
                    '水': 11,
                    '很': 12,
                    '不': 13,
                    '盡': 14,
                    '的': 15,
                    '表': 16,
                    '亞': 17,
                    '買': 18,
                    '一': 19,
                    'tank': 20,
                    '有': 21,
                    '些': 22,
                    'Black': 23,
                    'Tudor': 24,
                    '地': 25,
                    '言': 26,
                    '難': 27,
                    '卡': 28,
                    '男': 29,
                    '過': 30, 
                    '竞': 31, 
                    '赛': 32}

    def test_without_frequency_filter(self):
        # case I: no infrequent word filter , check exact value
        self.args.model_config['vocab_freq_cutoff'] = 0
        tokenizer = MultiLingualTokenizer(word_to_id=None, required_token_types=['CHAR', 'LETTERS'])
        build_vocab_from_dataset(datasets=['train', 'dev'], tokenizer=tokenizer,args=self.args)
        self.assertEqual(set(tokenizer.word_to_id.keys()), set(self.word_to_id_0.keys()))
        os.system("rm ./word_to_id.json")

    def test_single_frequency_filter(self):
        # case II: 90% infrequent word filter , check remaining words
        self.args.model_config['vocab_freq_cutoff'] = 0.9 
        tokenizer = MultiLingualTokenizer(word_to_id=None, required_token_types=['CHAR', 'LETTERS'])
        build_vocab_from_dataset(datasets=['train', 'dev'], tokenizer=tokenizer,args=self.args)
        self.assertEqual(len(tokenizer.word_to_id), 6)
        for w in tokenizer.word_to_id:
            self.assertIn(w, ['<OOV>', '<PAD>', '地', '亞', '卡', '一'])
        os.system("rm ./word_to_id.json")
        
    def test_increasing_frequency_filter(self):
        # case III: increasing % infrequent word filter , check vocab size
        tokenizer = MultiLingualTokenizer(word_to_id=None, required_token_types=['CHAR', 'LETTERS'])
        for i in range(1, 10):
            self.args.model_config['vocab_freq_cutoff'] = i / 10
            build_vocab_from_dataset(datasets=['train', 'dev'], tokenizer=tokenizer,args=self.args)
            acutal_vocab_size = len(tokenizer.word_to_id)
            expected_vocab_size = len(self.word_to_id_0) - int((len(self.word_to_id_0) - 2) * (i / 10))
            self.assertEqual(acutal_vocab_size, expected_vocab_size)
        os.system("rm ./word_to_id.json")