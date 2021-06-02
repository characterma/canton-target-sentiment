import unittest
import os, sys
import torch
sys.path.append("../src/")
from tokenizer import InternalTokenizer


class TestEndToEnd(unittest.TestCase):
    def test_freeze_bert_embeddings(self):
        os.chdir("../src/")
        os.system("python run.py --config_dir='../tests/test_e2e_samples/1/'")
        os.system("python run.py --config_dir='../tests/test_e2e_samples/2/'")

        state_path1 = '../tests/test_e2e_samples/1/model_state.pt'
        state_path2 = '../tests/test_e2e_samples/2/model_state.pt'
        state_dict1 = torch.load(state_path1, map_location="cpu")
        state_dict2 = torch.load(state_path2, map_location="cpu")

        for param_name in state_dict1.keys():
            if param_name.startswith("pretrained_model.embeddings.word_embeddings"):
                self.assertTrue(torch.equal(state_dict1[param_name], state_dict2[param_name]))

        self.assertTrue(not torch.equal(state_dict1['classifier.linear.weight'], state_dict2['classifier.linear.weight']))