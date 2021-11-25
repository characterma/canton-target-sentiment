import os
import unittest
import papermill as pm
import scrapbook as sb


task_to_config = {
    'sequence_classification': {
        'model': 'BERT_AVG',  
        'text_prepro': None, 
        'model_params': {'num_train_epochs': 5}, 
        'train_params': {'batch_size': 16}, 
        'eval_params': {'batch_size': 32}, 
    }, 
    'target_classification': {
        'model': 'TDBERT',  
        'text_prepro': None, 
        'model_params': {'num_train_epochs': 5}, 
        'train_params': {'batch_size': 16}, 
        'eval_params': {'batch_size': 32}, 
    },  
    'chinese_word_segmentation': {
        'model': 'BERT_CRF',  
        'text_prepro': None, 
        'model_params': {'num_train_epochs': 15}, 
        'train_params': {'batch_size': 16}, 
        'eval_params': {'batch_size': 32}, 
    }, 
}

src_dir = "../src"
target_macro_f1 = 1
tolerance = 0.1
device = 0
kernel_name = "python3"
notebook_path = "../notebooks/pipeline/train_new_model_with_specific_config.ipynb"
output_notebook = "../notebooks/pipeline/train_new_model_with_specific_config_tmp.ipynb"


class TestPipeline(unittest.TestCase):
    def test_pipeline_with_specific_model(self):
        model_dir = "../output/test_pipeline_with_specific_model_tmp" 
        for task, config in task_to_config.items():
            pm.execute_notebook(
                notebook_path,
                output_notebook,
                kernel_name=kernel_name,
                parameters=dict(
                    src_dir=src_dir,
                    task=task, 
                    device=device, 
                    model=config['model'],
                    text_prepro=config['text_prepro'],
                    model_params=config['model_params'],
                    train_params=config['train_params'],
                    eval_params=config['eval_params'],
                    model_dir=model_dir
                ),
            )
            results = sb.read_notebook(output_notebook).scraps.data_dict
            self.assertAlmostEqual(results["macro_f1"], target_macro_f1, delta=tolerance)

    # def test_pipeline_with_default_model(self):
    #     model_dir = "../output/test_pipeline_with_default_model_tmp" 
    #     for task, config in task_to_config.items():
    #         pm.execute_notebook(
    #             notebook_path,
    #             output_notebook,
    #             kernel_name=kernel_name,
    #             parameters=dict(
    #                 src_dir=src_dir,
    #                 task=task, 
    #                 device=device, 
    #                 model=None,
    #                 text_prepro=config['text_prepro'],
    #                 model_params=config['model_params'],
    #                 train_params=config['train_params'],
    #                 eval_params=config['eval_params'],
    #                 model_dir=model_dir
    #             ),
    #         )
    #         results = sb.read_notebook(output_notebook).scraps.data_dict
    #         self.assertAlmostEqual(results["macro_f1"], target_macro_f1, delta=tolerance)

    def tearDown(self):
        os.system(f"rm {output_notebook}")
        for output_folder in [
            '../output/test_pipeline_with_default_model_tmp',
            '../output/test_pipeline_with_specific_model_tmp'
        ]:
            os.system(f"rm -rf {output_folder}")