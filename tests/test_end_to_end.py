import unittest
import os
import sys
sys.path.append("../src/")


class TestEndToEnd(unittest.TestCase):
    task_models = [
                   'chinese_word_segmentation/CNN_CRF',
                   'chinese_word_segmentation/BERT_CRF', 
                   'sequence_classification/BERT_AVG', 
                   'sequence_classification/BERT_CLS', 
                   'sequence_classification/TEXT_CNN', 
                   'sequence_classification/TEXT_CNN_kd', 
                   'sequence_classification/TEXT_CNN_kd_dtd',
                   'target_classification/TDBERT', 
                   'target_classification/TGSAN', 
                   'target_classification/TGSAN2'
                   ]

    skip_onnx = [
                   'chinese_word_segmentation/CNN_CRF',
                   'chinese_word_segmentation/BERT_CRF', 
                   ]
    os.chdir("../src/")

    @classmethod
    def tearDownClass(cls):
        for task_model in cls.task_models:
            os.system(f"rm -rf ../config/examples/{task_model}/result")
            os.system(f"rm -rf ../config/examples/{task_model}/model")
            os.system(f"rm -rf ../config/examples/{task_model}/logs")
            os.system(f"rm ../config/examples/{task_model}/log")

    def test_models(self):
        for task_model in self.task_models:
            code = os.system(f"python run.py --config_dir='../config/examples/{task_model}'")
            self.assertEqual(code, 0, task_model)
            
        for task_model in self.task_models:
            code = os.system(f"python run.py --config_dir='../config/examples/{task_model}' --test_only")
            self.assertEqual(code, 0, task_model)

    def test_onnx(self):
        for task_model in self.task_models:
            if task_model not in self.skip_onnx:
                code = os.system(f"python build_onnx.py --config_dir='../config/examples/{task_model}'")
                self.assertEqual(code, 0, task_model)

                code = os.system(f"python optimize_onnx.py --config_dir='../config/examples/{task_model}'")
                self.assertEqual(code, 0, task_model)
