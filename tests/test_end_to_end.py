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
# #                    'sequence_classification/BERT_AVG_explain', 
                   'target_classification/TDBERT', 
                   'target_classification/TGSAN', 
                   'target_classification/TGSAN2'
                   ]

    def test_models(self):
        os.chdir("../src/")
        for task_model in self.task_models:
            code = os.system(f"python run.py --config_dir='../config/examples/{task_model}'")
            self.assertEqual(code, 0, task_model)
            
        for task_model in self.task_models:
            code = os.system(f"python run.py --config_dir='../config/examples/{task_model}' --test_only")
            self.assertEqual(code, 0, task_model)
            
#             if "explain" in task_model:
#                 code = os.system(f"python run.py --config_dir='../config/examples/{task_model}' --test_only --explain --faithfulness")
#                 self.assertEqual(code, 0, task_model)

    def tearDown(self):
        for task_model in self.task_models:
            os.system(f"rm -rf ../config/examples/{task_model}/result")
            os.system(f"rm -rf ../config/examples/{task_model}/model")
            os.system(f"rm -rf ../config/examples/{task_model}/logs")
            os.system(f"rm ../config/examples/{task_model}/log")
