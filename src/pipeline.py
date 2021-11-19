import torch 
import logging
from pathlib import Path

from label import get_label_to_id
from dataset import get_feature_class, get_dataset
from dataset.utils import get_model_inputs
from model import get_model
from trainer import evaluate, Trainer
from tokenizer import get_tokenizer
from utils import get_args, load_config, save_config, set_log_path


logger = logging.getLogger(__name__)
set_log_path("./")


class Pipeline:
    """
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
                 device=None
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
            self.args = get_args(config_dir=model_dir)
            self.args = load_config(self.args)
            if device is not None:
                self.args.device = device
            self.args.model_dir = Path(model_dir)
            self.task = self.args.run_config['task']
            self._initialize()
        else:
            assert(task is not None)
            logger.info(f"***** Model class is not provided for {task}. *****")
            if model is None:
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
            self.args = get_args(config_dir=config_dir)
            self.args = load_config(self.args)
            if device is not None:
                self.args.device = device
            self.task = task
        
            self.initialized = False
            self.tokenizer = None
            self.model = None
            self.feature_class = None
        
        self.train_dataset = None 
        self.dev_dataset = None 
        self.test_dataset = None 
 
    def train(self, 
              raw_data=None, 
              data_dir=None, 
              train_file=None, 
              dev_file=None, 
              output_dir=None, 
              model_params=None
             ):
        """
        Args:
            raw_data: list of dict.
            data_dir: str or path object.
            train_file: str
            dev_file: str
            output_dir: str or path object.
            model_params: dict.
        """
        # Over-writing args parameters
        if model_params is not None:
            self.args.model_config.update(model_params)
        if output_dir is not None:
            output_dir = Path(output_dir)
            self._overwrite_output_dir(output_dir)
        if data_dir is not None:
            self.args.data_dir = Path(data_dir)
        if train_file is not None:
            self.args.data_config['train'] = train_file
        if dev_file is not None:
            self.args.data_config['dev'] = dev_file
            
        self._initialize()

        # Prepare datasets.
        self.train_dataset = get_dataset(
            dataset="train", 
            tokenizer=self.tokenizer, 
            args=self.args
        )
        self.dev_dataset = get_dataset(
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
             raw_data=None,
             data_dir=None, 
             test_file=None
            ):
        """
        Args:
            raw_data: list of dict.
            data_dir: str or path object.
            test_file: str
        Returns:
            test_metrics: dict.
        """
        # Over-writing args parameters
        if data_dir is not None:
            self.args.data_dir = Path(data_dir)
        if test_file is not None:
            self.args.data_config['test'] = test_file
            
        # Prepare datasets.
        self.test_dataset = get_dataset(
            dataset="test", 
            tokenizer=self.tokenizer, 
            args=self.args
        )
        if self.train_dataset is not None:
            train_metrics = evaluate(
                model=self.model, 
                eval_dataset=self.train_dataset, 
                args=self.args
            )
        if self.dev_dataset is not None:
            dev_metrics = evaluate(
                model=self.model, 
                eval_dataset=self.dev_dataset, 
                args=self.args
            )
            
        # Start evaluation.
        test_metrics = evaluate(
            model=self.model, 
            eval_dataset=self.test_dataset, 
            args=self.args
        )
        return test_metrics

    def predict(self, data_dict):
        """
        Args:
            data_dict: dict.
        Returns:
            prediction: int or str.
        """
        
        # Preprocessing, tokenization, etc.
        feature_dict = self.feature_class(
            data_dict=data_dict, 
            tokenizer=self.tokenizer, 
            args=self.args, 
        ).feature_dict
        
        # Making batch.
        batch = dict()
        for col in feature_dict:
            batch[col] = feature_dict[col].unsqueeze(0).to(self.model.device)
            
        output = self.model(**batch)
        prediction_id = output["prediction"][0]
        prediction = self.args.label_to_id_inv[prediction_id]
        return prediction
    
    def _initialize(self):
        logger.info("***** Initializing pipeline *****")
        self.tokenizer = get_tokenizer(args=self.args)
        self.feature_class = get_feature_class(args=self.args)
        label_to_id, label_to_id_inv = get_label_to_id(self.tokenizer, self.args)
        self.args.label_to_id = label_to_id
        self.args.label_to_id_inv = label_to_id_inv
        self.model = get_model(args=self.args)
        self.initialized = True
    
    def _overwrite_output_dir(self, output_dir):
        output_dir = Path(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.args.output_dir = output_dir
        self.args.model_dir = output_dir / "model"
        self.args.result_dir = output_dir / "result"
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        if not os.path.exists(self.args.result_dir):
            os.makedirs(self.args.result_dir)
            
            
if __name__=="__main__":
    
#     # Train
#     task = "sequence_classification"
#     data_dir = "../data/datasets/internal/sequence_classification/post_sentiment"
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
    model_dir = "../config/examples/sequence_classification/BERT_CLS/model"
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