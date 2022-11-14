import os
import checklist
import numpy as np
from checklist.test_types import MFT, INV, DIR
from checklist.editor import Editor

from typing import Union, Optional
from collections import OrderedDict
from pathlib import Path
from shutil import copyfile

from perturbation import Perturbation
from utils import load_checklist_data
from reporter import ReportGenerator


class SentimentCheckList():
    def __init__(self, args):

        self.output_dir = Path(args.output_dir)
        self.output_config_dir = Path(args.output_config_dir)
        self.output_detail_dir = Path(args.output_detail_dir)
        self.args = args
        self.editor = Editor()
        self.tests = []
        self.perturbation = Perturbation
        self._load_resources_data()

    def _load_resources_data(self):

        resources = self.args.resources
        self.data_dir = Path(resources.pop("data_dir"))
        self.data_collections = {}
        for k, v in resources.items():
            if isinstance(v, str):
                self.data_collections[k] = load_checklist_data(self.data_dir / resources[k])
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, str):
                        keyname = f"{k}.{kk}"
                        self.data_collections[keyname] = load_checklist_data(self.data_dir / resources[k][kk])

    def _get_data(self, data_key, return_meta=False):
        if data_key in self.data_collections:
            return self.data_collections[data_key]
        else:
            datapath = os.path.join(self.data_dir, data_key)
            if os.path.exists(datapath):
                data = load_checklist_data(Path(datapath))
                if not return_meta:
                    data = [d["content"] for d in data]
                return data
            else:
                return []

    def test_with_template(self, template, inputs, label, test_name, test_capability, test_description, nsamples):
        ret = self.editor.template(template, **inputs, labels=label, save=True, nsamples=nsamples, remove_duplicates=True)
        test = MFT(ret.data,
                   labels=ret.labels,
                   name=test_name,
                   capability=test_capability,
                   description=test_description
                   )
        return test

    def test_with_raw_inputs(self, template, inputs, label, test_name, test_capability, test_description, nsamples):
        data = inputs["input_a"]

        if type(label) in [list, np.array, np.ndarray]:
            labels_ = label
        else:
            labels_ = [label]*len(data)

        if nsamples and nsamples < len(data):
            data = data[:nsamples]
            labels_ = labels_[:nsamples]

        test = MFT(data,
                   labels=labels_,
                   name=test_name,
                   capability=test_capability,
                   description=test_description
                   )
        return test

    def test_with_perturbation(self, data, perturb_func, inputs, test_name, test_capability, test_description, nsamples):

        # data: list of dict
        # t_data: list of dict
        test_data, meta_data = Perturbation.perturb(data, perturb_func, nsamples=nsamples, **inputs)

        if len(test_data) > 0:
            test = INV(data=meta_data,
                       name=test_name,
                       capability=test_capability,
                       description=test_description,
                       )
        else:
            test = False
        return test

    def init_report_dir(self):
        # mkdir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_config_dir, exist_ok=True)
        os.makedirs(self.output_detail_dir, exist_ok=True)

        # save run.yaml and checklist.yaml
        copyfile(self.args.config_dir / "run.yaml", self.output_config_dir / "run.yaml")
        copyfile(self.args.checklist_config_dir / "checklist.yaml", self.output_config_dir / "checklist.yaml")

    def run_mft_test(self, pred_wrapper):
        checklist_MFT = self.args.tc_mft
        reports = []

        for i, testcase in enumerate(checklist_MFT):
            test_name = testcase.get("name")
            capacity = testcase.get("capacity")
            operator = testcase.get("operator")
            template = testcase.get("template")
            inputs = testcase.get("inputs")
            label = testcase.get("label")
            nsamples = testcase.get("nsamples")

            operator_func = getattr(self, operator)
            inputs = {k: self._get_data(v, return_meta=True) for k, v in inputs.items()}

            if label is not None:
                labels = str(label)
            elif operator == "test_with_raw_inputs":
                try:
                    data = list(inputs.values())[0]
                    if "label" in data[0]:
                        labels = [str(d["label"]) for d in data]
                except Exception as e:
                    print(e)
            else:
                raise AssertionError("required variable: label is not set!")

            try:
                test = operator_func(template, inputs, labels, test_name, capacity, test_name, nsamples)
                test.run(pred_wrapper)
                report = ReportGenerator.generate_report_for_MFT(test, i+1, self.output_detail_dir)
                reports.extend(report)
            except Exception as e:
                print(e)

        return reports

    def run_inv_test(self, pred_wrapper):
        checklist_INV = self.args.tc_inv
        reports = []

        for i, testcase in enumerate(checklist_INV):
            test_name = testcase.get("name")
            capacity = testcase.get("capacity")
            operator = testcase.get("operator")
            inputs = testcase.get("inputs")
            label = str(testcase.get("label", 0))
            nsamples = testcase.get("nsamples")

            operator_func = getattr(self.perturbation, operator)
            data = self._get_data(inputs["data"], return_meta=True)
            data_alternatives = self._get_data(inputs["alternatives"]) if inputs.get("alternatives") else None
            data_replacement_dict = self._get_data(inputs["replacement_dict"]) if inputs.get("replacement_dict") else None
            data_keywords = self._get_data(inputs["keywords"]) if inputs.get("keywords") else None

            inputs = {
                "purturbation_ratio": inputs.get("purturbation_ratio", 0.3),
                "segmentation": inputs.get("segmentation", "word"),
                "n_samples": inputs.get("samples", 1),
                "api_url": inputs.get("NER_endpoint"),
                "target_type": inputs.get("NER_target_field"),
                "alternatives": data_alternatives,
                "replacement_dict": data_replacement_dict,
                "keywords": data_keywords,
            }

            test = self.test_with_perturbation(data, operator_func, inputs, test_name, capacity, test_name, nsamples)
            if test:
                test.run(pred_wrapper)
                report = ReportGenerator.generate_report_for_INV(test, i+1, self.output_detail_dir)
                reports.append(report)

        return reports

    def run_test(self, pred_wrapper):

        self.init_report_dir()
        self.tests = []
        summaries = {}
    
        # MFT
        if self.args.tc_mft:
            mft_reports = self.run_mft_test(pred_wrapper)
            summaries["MFT"] = mft_reports

        # INV
        if self.args.tc_inv:
            inv_reports = self.run_inv_test(pred_wrapper)
            summaries["INV"] = inv_reports

        # DIR

        # write summary
        ReportGenerator.save_summary(summaries, self.output_dir)

