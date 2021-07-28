import unittest
import os
import sys
import torch
sys.path.append("../src/")


class TestSetSeed(unittest.TestCase):
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
        for i in range(5, 8):

            os.system(f"rm -rf ../tests/test_end_to_end_samples/{i}/model")
            os.system(f"rm -rf ../tests/test_end_to_end_samples/{i}/result")
            os.system(f"rm -rf ../tests/test_end_to_end_samples/{i}/logs")
            os.system(f"rm ../tests/test_end_to_end_samples/{i}/log")
