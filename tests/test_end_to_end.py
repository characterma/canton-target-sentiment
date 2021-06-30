import unittest
import os, sys
import torch
from pathlib import Path 

sys.path.append("../src/")
from utils import load_config
from trainer import prediction_step
from dataset.target_classification import TargetClassificationFeature
from model import get_model
from label import get_label_to_id
from tokenizer import get_tokenizer
from collections import namedtuple
from dataset.utils import get_model_inputs


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


    def test_train_and_inference(self):
        os.chdir("../src/")

        os.system(f"python run.py --config_dir='../tests/test_end_to_end_samples/8/'")
        args = namedtuple('args', 'config_dir')
        args.config_dir = Path(f"../tests/test_end_to_end_samples/8")
        args = load_config(args)

        tokenizer = get_tokenizer(args=args)

        label_to_id, label_to_id_inv = get_label_to_id(tokenizer, args)
        args.label_to_id =  label_to_id
        args.label_to_id_inv = label_to_id_inv

        model = get_model(args=args)
        
        data_dict = {
            "content": "標題[心得] Tudor Black Bay 58 M79030B", 
            "target_locs": [[7, 12], [13, 18], [19, 22]]
        }

        data = TargetClassificationFeature(
            data_dict=data_dict,
            tokenizer=tokenizer,
            prepro_config=args.run_config['text_prepro'],
            max_length=args.model_config['max_length'],
            required_features=get_model_inputs(args),
            label_to_id=None,
        )

        batch = dict()
        for col in data.feature_dict:
            batch[col] = torch.stack([data.feature_dict[col]], dim=0)

        results = prediction_step(model, batch, args)
        self.assertTrue(results['prediction'][0]=='neutral')
        self.assertTrue(results['loss'] is None)

    def test_set_seed(self):
        os.chdir("../src/")

        # Test same seed
        os.system("python run.py --config_dir='../tests/test_end_to_end_samples/5/'")
        os.system("python run.py --config_dir='../tests/test_end_to_end_samples/6/'")

        model_path1 = '../tests/test_end_to_end_samples/5/model/model.pt'
        model_path2 = '../tests/test_end_to_end_samples/6/model/model.pt'
        state_dict1 = torch.load(model_path1).state_dict()
        state_dict2 = torch.load(model_path2).state_dict()
        for param_name in state_dict1.keys():
            self.assertTrue(
                torch.equal(state_dict1[param_name], state_dict2[param_name]), param_name
            )

        # Test different seeds
        os.system("python run.py --config_dir='../tests/test_end_to_end_samples/7/'")
        model_path3 = '../tests/test_end_to_end_samples/7/model/model.pt'
        state_dict3 = torch.load(model_path3).state_dict()
        has_diff = False
        for param_name in state_dict1.keys():
            if not torch.equal(state_dict1[param_name], state_dict3[param_name]):
                has_diff = True 
        self.assertTrue(has_diff)

    def tearDown(self):
        os.system("rm -rf ../tests/test_end_to_end_samples/1/model")
        os.system("rm -rf ../tests/test_end_to_end_samples/1/result")
        os.system("rm -rf ../tests/test_end_to_end_samples/2/model")
        os.system("rm -rf ../tests/test_end_to_end_samples/2/result")
        os.system("rm -rf ../tests/test_end_to_end_samples/3/model")
        os.system("rm -rf ../tests/test_end_to_end_samples/3/result")
        os.system("rm -rf ../tests/test_end_to_end_samples/4/model")
        os.system("rm -rf ../tests/test_end_to_end_samples/4/result")
        os.system("rm -rf ../tests/test_end_to_end_samples/5/model")
        os.system("rm -rf ../tests/test_end_to_end_samples/5/result")
        os.system("rm -rf ../tests/test_end_to_end_samples/6/model")
        os.system("rm -rf ../tests/test_end_to_end_samples/6/result")
        os.system("rm -rf ../tests/test_end_to_end_samples/7/model")
        os.system("rm -rf ../tests/test_end_to_end_samples/7/result")
        os.system("rm -rf ../tests/test_end_to_end_samples/8/model")
        os.system("rm -rf ../tests/test_end_to_end_samples/8/result")



