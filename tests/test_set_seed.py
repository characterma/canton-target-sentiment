import unittest
import os
import sys
import torch
from pathlib import Path, PurePath


class TestSetSeed(unittest.TestCase):

    test_dir = Path(PurePath(__file__).parent).resolve()
    src_dir = test_dir.parent / "nlp_pipeline"
    config_dir = test_dir.parent / "config"
    os.chdir(src_dir)

    def test_set_seed(self):
        sys.path.append(self.src_dir)
        os.system(f"python {self.src_dir}/run.py --config_dir='{self.test_dir}/test_end_to_end_samples/5/'")
        os.system(f"python {self.src_dir}/run.py --config_dir='{self.test_dir}/test_end_to_end_samples/6/'")

        model_path1 = f'{self.test_dir}/test_end_to_end_samples/5/model/model.pt'
        model_path2 = f'{self.test_dir}/test_end_to_end_samples/6/model/model.pt'
        state_dict1 = torch.load(model_path1)
        state_dict2 = torch.load(model_path2)
        for param_name in state_dict1.keys():
            self.assertTrue(
                torch.equal(state_dict1[param_name], state_dict2[param_name]), param_name
            )

        # Test different seeds
        os.system(f"python {self.src_dir}/run.py --config_dir='{self.test_dir}/test_end_to_end_samples/7/'")
        model_path3 = f'{self.test_dir}/test_end_to_end_samples/7/model/model.pt'
        state_dict3 = torch.load(model_path3)
        has_diff = False
        for param_name in state_dict1.keys():
            if not torch.equal(state_dict1[param_name], state_dict3[param_name]):
                has_diff = True
        self.assertTrue(has_diff)

    def tearDown(self):
        for i in range(5, 8):

            os.system(f"rm -rf {self.test_dir}/test_end_to_end_samples/{i}/model")
            os.system(f"rm -rf {self.test_dir}/test_end_to_end_samples/{i}/result")
            os.system(f"rm -rf {self.test_dir}/test_end_to_end_samples/{i}/logs")
            os.system(f"rm {self.test_dir}/test_end_to_end_samples/{i}/log")
