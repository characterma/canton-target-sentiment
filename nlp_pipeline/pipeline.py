import torch 
import logging
import os
import numpy as np
import pandas as pd
from pathlib import Path

from nlp_pipeline.label import get_label_to_id
from nlp_pipeline.dataset import get_feature_class, get_dataset
from nlp_pipeline.dataset.utils import get_model_inputs
from nlp_pipeline.model import get_model
from nlp_pipeline.trainer import evaluate, Trainer
from nlp_pipeline.tokenizer import get_tokenizer
from nlp_pipeline.explain import ExplainModel
from nlp_pipeline.explain.faithfulness import Comprehensiveness
from nlp_pipeline.utils import get_args, load_config, save_config, set_log_path
from nlp_pipeline.utils import combine_and_save_metrics, combine_and_save_statistics


logger = logging.getLogger(__name__)
set_log_path("./")


class Pipeline:
    """
    A high level pipeline for testing, training and prediction.
    TODO: support raw data input for get_dataset.
    
        1. __init__: Provided model_dir? if yes, load previous model, otherwise get ready to initialize new model.
            b. Provided model class? if yes, get ready to load new model, otherwise use default model class.
        2a. train: Initialize and train new model using provided datasets.
        2b. test: Test an existing model using provided datasets.
        3. predict: make prediction on a raw data.
    """
    def __init__(self, 
                 model_dir=None, 
                 task=None, 
                 model=None, 
                 device=None, 
                 text_prepro=None
                ):
        """
        Args:
            model_dir: str or path object.
            task: str
            model: str
            device: int or "cpu"
        """
        if model_dir is not None:
            logger.info(f"***** Existing model is provided. *****")
            logger.info("  Model directory = %s", str(model_dir))
            model_dir = Path(model_dir)
            self.args = get_args(config_dir=model_dir / "model")
            self.args = load_config(self.args)
            self.args.model_dir = model_dir / "model"
            if device is not None:
                self.args.device = device
            if text_prepro is not None:
                self.args.prepro_config.update(text_prepro)
            self.task = self.args.run_config['task']
            self._initialize()
        else:
            assert(task is not None)
            if model is None:
                logger.info(f"***** Model class is not specified for {task}. *****")
                if task=="sequence_classification":
                    default_model = "BERT_CLS"
                elif task=="target_classification":
                    default_model = "TDBERT"
                elif task=="chinese_word_segmentation":
                    default_model = "BERT_CRF"
                else:
                    raise(ValueError(f"Task {task} is not supported."))
                logger.info("  Default model = %s", default_model)
                config_dir = f"../config/examples/{task}/{default_model}"
            else:
                logger.info(f"***** Model class is specified for {task}. *****")
                logger.info("  Model = %s", model)
                config_dir = f"../config/examples/{task}/{model}"

            self.args = get_args(config_dir=config_dir)
            self.args = load_config(self.args)
            if device is not None:
                self.args.device = device
            if text_prepro is not None:
                self.args.prepro_config.update(text_prepro)
            self.task = task
        
            self.initialized = False
            self.tokenizer = None
            self.model = None
            self.feature_class = None
        
        self.train_dataset = None 
        self.dev_dataset = None 
        self.test_dataset = None 
 
    def train(self, 
              model_dir,
              train_raw_data=None,
              dev_raw_data=None, 
              data_dir=None, # data on disk
              train_file=None, 
              dev_file=None, 
              model_params=None, 
              train_params=None
             ):
        """
        Args:
            model_dir: str or path.
            train_raw_data: list of dict.
            dev_raw_data: list of dict.
            data_dir: str or path object.
            train_file: str
            dev_file: str
            model_dir: str or path object.
            model_params: dict.
        """
        assert((train_raw_data is not None and dev_raw_data is not None) or data_dir is not None)
        # Over-writing args parameters
        self._overwrite_model_dir(model_dir=model_dir)
        if model_params is not None:
            self.args.model_config.update(model_params)
        if train_params is not None:
            self.args.train_config.update(train_params)
        if data_dir is not None:
            self.args.data_dir = Path(data_dir)
        if train_file is not None:
            self.args.data_config['train'] = train_file
        if dev_file is not None:
            self.args.data_config['dev'] = dev_file
            
        self._initialize(train_raw_data=train_raw_data)
        # Prepare datasets.
        self.train_dataset = get_dataset(
            raw_data=train_raw_data,
            dataset="train", 
            tokenizer=self.tokenizer, 
            args=self.args
        )
        self.dev_dataset = get_dataset(
            raw_data=dev_raw_data,
            dataset="dev", 
            tokenizer=self.tokenizer, 
            args=self.args
        )
        
        # Start training.
        trainer = Trainer(
            model=self.model, 
            train_dataset=self.train_dataset, 
            dev_dataset=self.dev_dataset, 
            args=self.args
        )
        trainer.train()
        save_config(self.args)
        
    def test(self, 
             test_raw_data=None,
             data_dir=None, 
             test_file=None, 
             eval_params=None
            ):
        """
        Args:
            test_raw_data: list of dict.
            data_dir: str or path object.
            test_file: str
        Returns:
            test_metrics: dict.
        """
        assert(test_raw_data is not None or data_dir is not None)
        # Over-writing args parameters
        if data_dir is not None:
            self.args.data_dir = Path(data_dir)
        if test_file is not None:
            self.args.data_config['test'] = test_file
        if eval_params is not None:
            self.args.eval_config.update(eval_params)
            # print(self.args.eval_config)
            
        # Prepare datasets.
        self.test_dataset = get_dataset(
            raw_data=test_raw_data, 
            dataset="test", 
            tokenizer=self.tokenizer, 
            args=self.args
        )

        # Start evaluation.
        test_metrics = evaluate(
            model=self.model, 
            eval_dataset=self.test_dataset, 
            args=self.args
        )
        combine_and_save_metrics(metrics=[test_metrics], args=self.args, suffix="pipeline_test")
        combine_and_save_statistics(datasets=[self.test_dataset], args=self.args, suffix="pipeline_test")
        return test_metrics

    def predict(self, data_dict):
        """
        Args:
            data_dict: dict.
        Returns:
            prediction: int or str.
        """
        
        # Preprocessing, tokenization, etc.
        self.model.eval()
        feature_dict = self.feature_class(
            data_dict=data_dict, 
            tokenizer=self.tokenizer, 
            args=self.args, 
            padding=False
        ).feature_dict
        
        # Making batch.
        batch = dict()
        for col in feature_dict:
            batch[col] = feature_dict[col].unsqueeze(0).to(self.args.device)
            
        output = self.model(**batch)
        results = dict(prediction_id=None, prediction=None, logits=None)

        if isinstance(output['prediction'], torch.Tensor):
            prediction = output['prediction'].tolist()[0]
            logits = output['logits'].tolist()[0]
        else:
            prediction = output['prediction'][0]
            logits = output['logits'][0]

        if isinstance(prediction, list):
            results["prediction_id"] = prediction
            results["prediction"] = list(map(lambda y: self.args.label_to_id_inv[y], prediction))
            results['logits'] = logits
        else:
            results["prediction_id"] = prediction
            results["prediction"] = self.args.label_to_id_inv[prediction]
            results['logits'] = logits
        return results

    def explain(self, data_dict, method, enable_faithfulness=False, **kwargs):
        assert(self.task == "sequence_classification")
        config = {'method': method}
        config.update(kwargs)
        explain_model = ExplainModel(
            model=self.model, config=config
        )

        self.model.eval()
        feature = self.feature_class(
            data_dict=data_dict, 
            tokenizer=self.tokenizer, 
            args=self.args, 
            diagnosis=True,
            padding=False
        )

        feature_dict = feature.feature_dict
        diagnosis_dict = feature.diagnosis_dict
        tokens_encoded = feature.tokens_encoded
        
        # Making batch.
        batch = dict()
        for col in feature_dict:
            batch[col] = feature_dict[col].unsqueeze(0).to(self.args.device)

        scores, attr_target, attr_target_prob = explain_model(
            inputs=batch, 
            target=None, 
            pad_token_id=self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') else None, 
            sep_token_id=self.tokenizer.sep_token_id if hasattr(self.tokenizer, 'sep_token_id') else None, 
            cls_token_id=self.tokenizer.cls_token_id if hasattr(self.tokenizer, 'cls_token_id') else None
        )

        if enable_faithfulness:
            faithfulness = np.mean(
                Comprehensiveness(
                    model=self.model, 
                    inputs=batch, 
                    scores=scores, 
                    unk_token_id=self.tokenizer.unk_token_id, 
                    pad_token_id=self.tokenizer.pad_token_id
                ).comprehensiveness
            )
        else:
            faithfulness = None

        scores = scores.tolist()[0]
        tokens = diagnosis_dict['tokens']
        assert(len(scores)==len(tokens))

        return tokens, scores, attr_target, attr_target_prob, tokens_encoded, faithfulness

    def _initialize(self, train_raw_data=None):
        logger.info("***** Initializing pipeline *****")
        self.tokenizer = get_tokenizer(args=self.args)
        self.feature_class = get_feature_class(args=self.args)
        label_to_id, label_to_id_inv = get_label_to_id(
            self.tokenizer, 
            self.args, 
            raw_data=train_raw_data
        )
        self.args.label_to_id = label_to_id
        self.args.label_to_id_inv = label_to_id_inv
        self.model = get_model(args=self.args)
        self.initialized = True
    
    def _overwrite_model_dir(self, model_dir):
        model_dir = Path(model_dir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.args.output_dir = model_dir
        self.args.model_dir = model_dir / "model"
        self.args.result_dir = model_dir / "result"
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        if not os.path.exists(self.args.result_dir):
            os.makedirs(self.args.result_dir)
            
            
if __name__=="__main__":
    
#     # Train
#     task = "sequence_classification"
#     data_dir = "../data/datasets/internal/sequence_classification/post_sentiment"
#     model_dir = "../output/test_pipeline"
#     train_file = "train_sample.json"
#     dev_file = "train_sample.json"
#     test_file = "train_sample.json"
#     device = 0
#     model_params = {
#         'num_train_epochs': 3,
#         "tokenizer_name": "hfl/chinese-macbert-base",
#         "pretrained_lm": "hfl/chinese-macbert-base"
#     }
    
#     pipeline = Pipeline(
#         task=task, 
#         device=device
#     )
#     pipeline.train(
#         model_dir=model_dir,
#         data_dir=data_dir, 
#         train_file=train_file, 
#         dev_file=dev_file, 
#         model_params=model_params
#     )

#     test_metrics = pipeline.test(
#         data_dir=data_dir, 
#         test_file=test_file
#     )
    
#     print(test_metrics)
    
    # Test only
    model_dir = "../output/test_pipeline"
    data_dir = "../data/datasets/internal/sequence_classification/post_sentiment"
    test_file = "train_sample.json"
    device = 0
    
    pipeline = Pipeline(
        model_dir=model_dir, 
        device=device
    )
    test_metrics = pipeline.test(
        data_dir=data_dir, 
        test_file=test_file
    )
    print(test_metrics)
    
    # Prediction
    x = {'content': "不过，随着乐视出现资金问题，烧钱严重的乐视体育全面收缩，接连丢掉中超、亚冠等赛事版权，在香港地区仅剩下2017-18赛季英超和NBA两项赛事直播权，此外，乐视体育香港也放弃了\"独播\"战略，对英超版权进行了分销，此举一方面可回笼部分资金，却对会员收入方面带来一定影响。"}
    print(pipeline.predict(x))