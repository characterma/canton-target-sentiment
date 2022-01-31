import os
import unittest
import papermill as pm
import scrapbook as sb


task_to_config = {
    'sequence_classification': {
        'BERT_AVG': {
            'text_prepro': None, 
            'model_params': {'num_train_epochs': 10}, 
            'train_params': {'batch_size': 16}, 
            'eval_params': {'batch_size': 32}, 
            'target_macro_f1': 1
        }, 
        'BERT_CLS': {
            'text_prepro': None, 
            'model_params': {'num_train_epochs': 10}, 
            'train_params': {'batch_size': 16}, 
            'eval_params': {'batch_size': 32}, 
            'target_macro_f1': 1
        }, 
        'TEXT_CNN': {
            'text_prepro': None, 
            'model_params': {'num_train_epochs': 25}, 
            'train_params': {'batch_size': 16}, 
            'eval_params': {'batch_size': 32}, 
            'target_macro_f1': 1
        }, 
    }, 
    'target_classification': {
        'TDBERT': {
            'text_prepro': None, 
            'model_params': {'num_train_epochs': 10}, 
            'train_params': {'batch_size': 16}, 
            'eval_params': {'batch_size': 32}, 
            'target_macro_f1': 1
        }, 
        'TGSAN': {
            'text_prepro': None, 
            'model_params': {'num_train_epochs': 50}, 
            'train_params': {'batch_size': 16}, 
            'eval_params': {'batch_size': 32}, 
            'target_macro_f1': 1
        }, 
        # 'TGSAN2': {
        #     'text_prepro': None, 
        #     'model_params': {'num_train_epochs': 100}, 
        #     'train_params': {'batch_size': 16}, 
        #     'eval_params': {'batch_size': 32}, 
        #     'target_macro_f1': 1
        # }
    },  
    'chinese_word_segmentation': {
        'BERT_CRF': {
            'text_prepro': None, 
            'model_params': {'num_train_epochs': 15}, 
            'train_params': {'batch_size': 16}, 
            'eval_params': {'batch_size': 32}, 
            'target_macro_f1': 1
        }, 
        'CNN_CRF': {
            'text_prepro': None, 
            'model_params': {'num_train_epochs': 100}, 
            'train_params': {'batch_size': 16}, 
            'eval_params': {'batch_size': 32}, 
            'target_macro_f1': 0.65
        }, 
    }, 
}

src_dir = "../src"
tolerance = 0.05
device = 0
kernel_name = "python3"
notebook_path = "../notebooks/pipeline/train_new_model_with_specific_config.ipynb"
output_notebook = "../notebooks/pipeline/train_new_model_with_specific_config_tmp.ipynb"


class TestPipeline(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        os.system(f"rm {output_notebook}")
        for output_folder in [
            '../output/test_pipeline_with_default_model_tmp',
            '../output/test_pipeline_with_specific_model_tmp'
        ]:
            os.system(f"rm -rf {output_folder}")
            
    def test_pipeline_with_specific_model(self):
        model_dir = "../output/test_pipeline_with_specific_model_tmp" 
        for task, model_to_config in task_to_config.items():
            for model, config in model_to_config.items():
                pm.execute_notebook(
                    notebook_path,
                    output_notebook,
                    kernel_name=kernel_name,
                    parameters=dict(
                        src_dir=src_dir,
                        task=task, 
                        device=device, 
                        model=model, 
                        text_prepro=config['text_prepro'],
                        model_params=config['model_params'],
                        train_params=config['train_params'],
                        eval_params=config['eval_params'],
                        model_dir=model_dir
                    ),
                )
                results = sb.read_notebook(output_notebook).scraps.data_dict
                self.assertAlmostEqual(
                    results["macro_f1"], 
                    config['target_macro_f1'], 
                    delta=tolerance, 
                    msg=model
                )