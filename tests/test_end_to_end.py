import unittest
import os
import sys
sys.path.append("../src/")


class TestEndToEnd(unittest.TestCase):
    task_models = ['chinese_word_segmentation/CNN_CRF',
                   'chinese_word_segmentation/BERT_CRF', 
                   'sequence_classification/BERT_AVG', 
                   'sequence_classification/TEXT_CNN', 
                   'target_classification/TDBERT', 
                   'target_classification/TGSAN', 
                   'target_classification/TGSAN2']

    def test_models(self):
        os.chdir("../src/")
        for task_model in self.task_models:
            code = os.system(f"python run.py --config_dir='../config/examples/{task_model}'")
            self.assertEqual(code, 0)

    def tearDown(self):
        for task_model in self.task_models:
            os.system(f"rm -rf ../config/examples/{task_model}/result")
            os.system(f"rm -rf ../config/examples/{task_model}/model")
            os.system(f"rm -rf ../config/examples/{task_model}/logs")
            os.system(f"rm ../config/examples/{task_model}/log")
