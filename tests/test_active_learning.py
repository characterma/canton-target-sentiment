import unittest
import os
import sys
from pathlib import Path, PurePath
import pathlib as pl

class TestActiveLearning(unittest.TestCase):
    test_dir = Path(PurePath(__file__).parent).resolve()
    al_data_dir = test_dir.parent / "output/active_learning/output"
    al_result_dir = test_dir.parent / "output/active_learning/result"
    src_dir = test_dir.parent / "nlp_pipeline"
    config_dir = test_dir.parent / "config"
    
    task_models = [
        "sequence_classification/BERT_AVG_AL",
        "sequence_classification/BERT_AVG_AL_EXP",
    ]

    def assertIsFile(self, path):
        if not pl.Path(path).resolve().is_file():
            raise AssertionError("File does not exist: %s" % str(path))

    @classmethod
    def tearDownClass(cls):
        os.system(f"rm {cls.al_data_dir}/*.json")
        os.system(f"rm {cls.al_result_dir}/*.pkl")

        for task_model in cls.task_models:
            os.system(f"rm -rf {cls.config_dir}/examples/{task_model}/result")
            os.system(f"rm -rf {cls.config_dir}/examples/{task_model}/model")
            os.system(f"rm -rf {cls.config_dir}/examples/{task_model}/logs")
            os.system(f"rm {cls.config_dir}/examples/{task_model}/log")

    def test_models(self):
        for task_model in self.task_models:
            code = os.system(f"python {self.src_dir}/run.py --config_dir='{self.config_dir}/examples/{task_model}'")
            self.assertEqual(code, 0, task_model)

            self.assertIsFile(self.al_data_dir / "active_learning_queried_data_0.json")
            if task_model == "sequence_classification/BERT_AVG_AL_EXP":
                self.assertIsFile(self.al_result_dir / "result.pkl")
