import unittest
import os
import sys
from pathlib import Path, PurePath


class TestEndToEnd(unittest.TestCase):
    task_models = [
                   'chinese_word_segmentation/CNN_CRF',
                   'chinese_word_segmentation/BERT_CRF', 
                   'sequence_classification/BERT_CLS', 
                   'sequence_classification/BERT_AVG', 
                   'sequence_classification/BERT_CLS_optim_tricks', 
                   'sequence_classification/TEXT_CNN', 
                   'sequence_classification/TEXT_CNN_kd', 
                   'sequence_classification/TEXT_CNN_kd_dtd',
                   'target_classification/TDBERT', 
                   'target_classification/TGSAN', 
                   'target_classification/TGSAN2',
                   'topic_classification/BERT_CLS',
                   'topic_classification/BERT_AVG',
                   ]

    skip_onnx = [
                   'chinese_word_segmentation/CNN_CRF',
                   'chinese_word_segmentation/BERT_CRF', 
                #    'target_classification/TGSAN',
                   'target_classification/TGSAN2'
                   ]
    skip_jit = [
                   'chinese_word_segmentation/CNN_CRF',
                   'chinese_word_segmentation/BERT_CRF', 
                    # 'target_classification/TGSAN',
                    # 'target_classification/TGSAN2'
    ]
    test_dir = Path(PurePath(__file__).parent).resolve()
    src_dir = test_dir.parent / "nlp_pipeline"
    config_dir = test_dir.parent / "config"
    # os.chdir(src_dir)

    @classmethod
    def tearDownClass(cls):
        for task_model in cls.task_models:
            os.system(f"rm -rf {cls.config_dir}/examples/{task_model}/result")
            os.system(f"rm -rf {cls.config_dir}/examples/{task_model}/model")
            os.system(f"rm -rf {cls.config_dir}/examples/{task_model}/logs")
            os.system(f"rm {cls.config_dir}/examples/{task_model}/log")

    def test_models(self):
        for task_model in self.task_models:
            code = os.system(f"python {self.src_dir}/run.py --config_dir='{self.config_dir}/examples/{task_model}'")
            self.assertEqual(code, 0, task_model)

            code = os.system(f"python {self.src_dir}/run.py --config_dir='{self.config_dir}/examples/{task_model}' --test_only")
            self.assertEqual(code, 0, task_model)

            if task_model not in self.skip_onnx:
                code = os.system(f"python {self.src_dir}/build_onnx.py --config_dir='{self.config_dir}/examples/{task_model}'")
                self.assertEqual(code, 0, task_model)

                code = os.system(f"python {self.src_dir}/optimize_onnx.py --config_dir='{self.config_dir}/examples/{task_model}'")
                self.assertEqual(code, 0, task_model)

            if task_model not in self.skip_jit:
                code = os.system(f"python {self.src_dir}/build_jit_trace.py --config_dir='{self.config_dir}/examples/{task_model}'")
                self.assertEqual(code, 0, task_model)
