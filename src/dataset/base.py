from preprocess import Preprocessor
from dataset.utils import get_model_inputs
import torch
import json
import abc
import logging
from tqdm import tqdm
import pandas as pd


logger = logging.getLogger(__name__)


class NLPDataset:
    def __init__(self, feature_class, dataset, tokenizer, args, raw_data=None):
        self.args = args
        self.dataset = dataset
        self.tokenizer = tokenizer

        self.required_features = get_model_inputs(args=args)
        self.feature_class = feature_class

        self.features = []
        self.diagnosis = []
        self.diagnosis_df = None
        self.skipped_indexs = []

        self.load_data(raw_data=raw_data)
        self.create_diagnosis()

    def load_data(self, raw_data=None):
        data_path = self.args.data_dir / self.args.data_config[self.dataset]
        logger.info("***** Loading data *****")
        
        if raw_data is not None:
            logger.info("  Raw data is provided.")
        else:
            logger.info("  Data path = %s", str(data_path))
            raw_data = json.load(open(data_path, "r"))

        for idx, data_dict in tqdm(enumerate(raw_data)):
            diagnosis_dict = {"idx": idx}
            fea = self.feature_class(
                data_dict=data_dict,
                tokenizer=self.tokenizer,
                args=self.args,
                diagnosis=True,
            )

            if fea.feature_dict is not None:
                self.features.append(fea.feature_dict)
            else:
                self.skipped_indexs.append(idx)

            diagnosis_dict.update(fea.diagnosis_dict)
            self.diagnosis.append(diagnosis_dict)
        logger.info("  Loaded samples = %d", len(self.features))

    def create_diagnosis(self):
        self.diagnosis_df = pd.DataFrame(data=self.diagnosis)
        self.diagnosis_df["dataset"] = self.dataset

    def get_data_analysis(self):
        statistics = {"dataset": self.dataset}
        statistics["total_samples"] = len(self.diagnosis)
        statistics["loaded_samples"] = len(self.features)
        return statistics

    def insert_skipped_samples(self, elements, value=None):
        for idx in self.skipped_indexs:
            elements.insert(idx, value)

    def insert_predictions(self, predictions):
        if len(predictions) != len(self.diagnosis):
            self.insert_skipped_samples(predictions)
        self.diagnosis_df["prediction"] = predictions

    def insert_diagnosis_column(self, values, name, update=False):
        if update and name in self.diagnosis_df:
            self.diagnosis_df.drop(columns=[name], inplace=True)

        if len(values) != len(self.diagnosis):
            self.insert_skipped_samples(values)
        self.diagnosis_df[name] = values

    def add_feature(self, name, values):
        # print(len(values), len(self.features))
        self.insert_skipped_samples(self.features, value=None)
        assert len(values) == len(self.features)
        features = []
        for x, y in zip(values, self.features):
            if x is not None and y is not None:
                y[name] = torch.tensor(x)
                features.append(y)
        self.features = features

    def __getitem__(self, index):
        return self.features[index]

    def __len__(self):
        return len(self.features)


class NLPFeature(abc.ABC):
    def __init__(self, data_dict, tokenizer, args, diagnosis=False, padding="max_length"):
        self.succeeded = True
        self.msg = ""
        self.padding = padding

        required_features = get_model_inputs(args)
        prepro_config = args.prepro_config

        preprocessor = Preprocessor(data_dict=data_dict, steps=prepro_config["steps"])

        self.feature_dict, self.diagnosis_dict = self.get_feature(
            data_dict=preprocessor.data_dict,
            tokenizer=tokenizer,
            required_features=required_features,
            args=args,
            diagnosis=diagnosis,
        )

    @abc.abstractmethod
    def get_feature(self, data_dict, tokenizer, required_features, args, diagnosis):
        return NotImplemented
