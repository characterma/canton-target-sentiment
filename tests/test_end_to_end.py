import unittest
import os, sys
import torch
sys.path.append("../src/")


class TestEndToEnd(unittest.TestCase):
    def test_freeze_bert_embeddings(self):
        os.chdir("../src/")
        os.system("python run.py --config_dir='../tests/test_end_to_end_samples/1/'")
        os.system("python run.py --config_dir='../tests/test_end_to_end_samples/2/'")

        model_path1 = '../tests/test_end_to_end_samples/1/model/model.pt'
        model_path2 = '../tests/test_end_to_end_samples/2/model/model.pt'
        state_dict1 = torch.load(model_path1).state_dict()
        state_dict2 = torch.load(model_path2).state_dict()

        for param_name in state_dict1.keys():
            if param_name.startswith("pretrained_model.embeddings"):
                self.assertTrue(torch.equal(state_dict1[param_name], state_dict2[param_name]))

        self.assertTrue(not torch.equal(state_dict1['classifier.linear.weight'], state_dict2['classifier.linear.weight']))
        os.system("rm -rf ../tests/test_end_to_end_samples/1/model")
        os.system("rm -rf ../tests/test_end_to_end_samples/1/result")
        os.system("rm -rf ../tests/test_end_to_end_samples/2/model")
        os.system("rm -rf ../tests/test_end_to_end_samples/2/result")

    def test_freeze_non_bert_embeddings(self):
        os.chdir("../src/")
        os.system("python run.py --config_dir='../tests/test_end_to_end_samples/3/'")
        os.system("python run.py --config_dir='../tests/test_end_to_end_samples/4/'")

        model_path1 = '../tests/test_end_to_end_samples/3/model/model.pt'
        model_path2 = '../tests/test_end_to_end_samples/4/model/model.pt'
        state_dict1 = torch.load(model_path1).state_dict()
        state_dict2 = torch.load(model_path2).state_dict()

        for param_name in state_dict1.keys():
            if param_name.startswith("embed."):
                self.assertTrue(torch.equal(state_dict1[param_name], state_dict2[param_name]), param_name)

        self.assertTrue(not torch.equal(state_dict1['fc.weight'], state_dict2['fc.weight']))
        os.system("rm -rf ../tests/test_end_to_end_samples/3/model")
        os.system("rm -rf ../tests/test_end_to_end_samples/3/result")
        os.system("rm -rf ../tests/test_end_to_end_samples/4/model")
        os.system("rm -rf ../tests/test_end_to_end_samples/4/result")

    def test_set_seed(self):
        os.chdir("../src/")

        # Test different seed
        os.system("python run.py --config_dir='../tests/test_end_to_end_samples/5/'")
        os.system("python run.py --config_dir='../tests/test_end_to_end_samples/6/'")

        model_path1 = '../tests/test_end_to_end_samples/5/model/model.pt'
        model_path2 = '../tests/test_end_to_end_samples/6/model/model.pt'
        state_dict1 = torch.load(model_path1).state_dict()
        state_dict2 = torch.load(model_path2).state_dict()
        has_diff = False
        for param_name in state_dict1.keys():
            if not torch.equal(state_dict1[param_name], state_dict2[param_name]):
                has_diff = True 
        self.assertTrue(has_diff)

        os.system("rm -rf ../tests/test_end_to_end_samples/5/model")
        os.system("rm -rf ../tests/test_end_to_end_samples/5/result")
        os.system("rm -rf ../tests/test_end_to_end_samples/6/model")
        os.system("rm -rf ../tests/test_end_to_end_samples/6/result")

        # Test same seed
        os.system("python run.py --config_dir='../tests/test_end_to_end_samples/5/'")
        model_path3 = '../tests/test_end_to_end_samples/5/model/model.pt'
        state_dict3 = torch.load(model_path3).state_dict()
        for param_name in state_dict1.keys():
            self.assertTrue(torch.equal(state_dict1[param_name], state_dict3[param_name]), param_name)
        os.system("rm -rf ../tests/test_end_to_end_samples/5/model")
        os.system("rm -rf ../tests/test_end_to_end_samples/5/result")


