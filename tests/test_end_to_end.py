import unittest
import os, sys
import torch
from pathlib import Path 

sys.path.append("../src/")
from dataset import TargetDependentExample
from utils import load_yaml
from transformers import BertTokenizerFast


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

    def test_inference(self):
        os.chdir("../src/")
        sample_id = 7
        os.system(f"python run.py --config_dir='../tests/test_end_to_end_samples/{sample_id}/'")

        # load configs and models
        run_config = load_yaml(
            Path(f"../tests/test_end_to_end_samples/{sample_id}/run.yaml")
        )
        model_config = load_yaml(
            Path(f"../tests/test_end_to_end_samples/{sample_id}/model.yaml")
        )
        model_class = run_config['train']['model_class']
        model_path = f'../tests/test_end_to_end_samples/{sample_id}/model/model.pt'
        model = torch.load(model_path)
        tokenizer = BertTokenizerFast.from_pretrained(model_config[model_class]['pretrained_lm'])

        # raw data without label
        data_dict = {
            'content': "#仪式感不能少没有卡地亚， 🔥浪琴，但是我有阿玛尼，“我愿意把星辰银河都送给你”别说人间不值得 你最值得！", 
            'target_locs': [[15, 17]]
        }

        # prepare feature
        data = TargetDependentExample(
            data_dict=data_dict,
            tokenizer=tokenizer,
            prepro_config=run_config['text_prepro'],
            max_length=model_config[model_class]['max_length'],
            required_features=model.INPUT,
            label_to_id=None,
        )

        # make batch
        batch = dict()
        for col in data.feature_dict:
            batch[col] = torch.stack([data.feature_dict[col]], dim=0)

        # predict
        model.eval()
        with torch.no_grad():
            inputs = dict()
            for col in batch:
                inputs[col] = batch[col].to(run_config['device']).long()
            x = model(
                **inputs,
            )

        loss = x[0]
        logits = x[1]
        self.assertTrue(loss is None, "Loss is not expected.")
        self.assertTrue(logits is not None)



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


