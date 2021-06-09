import unittest
import os, sys
import torch
from pathlib import Path 

sys.path.append("../src/")
from dataset import TargetDependentExample
from utils import load_config
from trainer import prediction_step
from run import init_model, init_tokenizer
from collections import namedtuple


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

    def test_train_and_inference(self):
        os.chdir("../src/")
        sample_id = 7
        os.system(f"python run.py --config_dir='../tests/test_end_to_end_samples/{sample_id}/'")

        args = namedtuple('args', 'config_dir')
        args.config_dir = Path(f"../tests/test_end_to_end_samples/{sample_id}")
        args = load_config(args)

        tokenizer = init_tokenizer(args=args)
        model = init_model(args=args)

        data_dict = {
            "content": "標題[心得] Tudor Black Bay 58 M79030B", 
            "target_locs": [[7, 12], [13, 18], [19, 22]]
        }

        data = TargetDependentExample(
            data_dict=data_dict,
            tokenizer=tokenizer,
            prepro_config=args.run_config['text_prepro'],
            max_length=args.model_config['max_length'],
            required_features=model.INPUT,
            label_to_id=None,
        )

        batch = dict()
        for col in data.feature_dict:
            batch[col] = torch.stack([data.feature_dict[col]], dim=0)

        results = prediction_step(model, batch, args)

        self.assertTrue(results['sentiment'][0]=='neutral')
        self.assertTrue(results['loss'] is None)
        os.system(f"rm -rf ../tests/test_end_to_end_samples/{sample_id}/model")
        os.system(f"rm -rf ../tests/test_end_to_end_samples/{sample_id}/result")


    # def test_set_seed(self):
    #     os.chdir("../src/")

    #     # Test different seed
    #     os.system("python run.py --config_dir='../tests/test_end_to_end_samples/5/'")
    #     os.system("python run.py --config_dir='../tests/test_end_to_end_samples/6/'")

    #     model_path1 = '../tests/test_end_to_end_samples/5/model/model.pt'
    #     model_path2 = '../tests/test_end_to_end_samples/6/model/model.pt'
    #     state_dict1 = torch.load(model_path1).state_dict()
    #     state_dict2 = torch.load(model_path2).state_dict()
    #     has_diff = False
    #     for param_name in state_dict1.keys():
    #         if not torch.equal(state_dict1[param_name], state_dict2[param_name]):
    #             has_diff = True 
    #     self.assertTrue(has_diff)

    #     os.system("rm -rf ../tests/test_end_to_end_samples/5/model")
    #     os.system("rm -rf ../tests/test_end_to_end_samples/5/result")
    #     os.system("rm -rf ../tests/test_end_to_end_samples/6/model")
    #     os.system("rm -rf ../tests/test_end_to_end_samples/6/result")

    #     # Test same seed
    #     os.system("python run.py --config_dir='../tests/test_end_to_end_samples/5/'")
    #     model_path3 = '../tests/test_end_to_end_samples/5/model/model.pt'
    #     state_dict3 = torch.load(model_path3).state_dict()
    #     for param_name in state_dict1.keys():
    #         self.assertTrue(torch.equal(state_dict1[param_name], state_dict3[param_name]), param_name)
    #     os.system("rm -rf ../tests/test_end_to_end_samples/5/model")
    #     os.system("rm -rf ../tests/test_end_to_end_samples/5/result")


