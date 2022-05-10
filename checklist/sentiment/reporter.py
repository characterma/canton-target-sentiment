import re
import numpy as np
import pandas as pd
from collections import OrderedDict


class ReportGenerator:

    @staticmethod
    def save_summary(reports, output_dir):

        writer = pd.ExcelWriter(output_dir / "summary.xlsx", engine="xlsxwriter")

        for k, v in reports.items():

            df = pd.DataFrame(v)
            cols = list(df.columns)
            cols.remove("filename")
            cols = cols + ["filename"]
            df = df[cols]
            df.to_excel(writer, index=False, sheet_name=k)

        writer.save()

    @staticmethod
    def generate_report_for_MFT(test, k, output_dir):

        def _breakdown(predictions):
            pred_counts = predictions.value_counts()
            pred_ratio = pred_counts / len(predictions)
            return pred_ratio.to_dict(OrderedDict)

        test_report = OrderedDict()
        test_name = test.name
        xs = test.data
        preds = test.results.preds
        labels = test.labels

        # model predictions
        df = pd.DataFrame()
        df["text"] = xs
        df["label"] = np.array(labels, dtype=str)
        df["prediction"] = np.array(preds, dtype=str)

        filename = "MFT-{}-predictions-{}.xlsx".format(str(k).zfill(2), re.sub(r"[+\- \/]+", "_", test_name.lower()))
        df.to_excel(output_dir / filename, index=False)

        # error rates
        test_reports = []
        test_report["test name"] = str(test_name)

        for lbl, gp in df.groupby("label"):

            test_report["expected label"] = lbl
            test_report["number of support"] = len(gp)

            # overall error rate
            acc = sum(gp["label"] == gp["prediction"]) / len(gp)
            test_report["error rate"] = 1 - acc

            # breakdown
            breakdown = _breakdown(gp["prediction"])
            test_report.update(breakdown)
            
            test_report["filename"] = filename

            test_reports.append(test_report.copy())
 
        return test_reports

    @staticmethod
    def generate_report_for_INV(test, k, output_dir):

        test_report = OrderedDict()
        test_summary = OrderedDict()
        test_name = test.name
        xs = test.data
        preds = test.results.preds
        labels = test.labels

        case_ids = []
        original_text = []
        perturbed_text = []
        original_pred = []
        perturbed_pred = []
        case_labels = []

        for i, (x, pred) in enumerate(zip(xs, preds)):
            x_b = x[1:]
            p_b = pred[1:]
            x_a = [x[0]] * len(p_b)
            p_a = [pred[0]] * len(p_b)
            case_id = [i] * len(p_b)

            case_ids.extend(case_id)
            original_text.extend(x_a)
            perturbed_text.extend(x_b)
            original_pred.extend(p_a)
            perturbed_pred.extend(p_b)
            case_labels.extend([-1] * len(p_b))
            # case_labels.extend([labels[i]] * len(p_b))
            
        test_report["case_id"] = case_ids
        test_report["original_text"] = original_text
        test_report["perturbed_text"] = perturbed_text
        test_report["manual_label"] = case_labels
        test_report["original_pred"] = original_pred
        test_report["perturbed_pred"] = perturbed_pred

        df = pd.DataFrame.from_dict(test_report)
        filename = "INV-{}-predictions-{}.xlsx".format(str(k).zfill(2), re.sub(r"[+\- \/]+", "_", test_name.lower()))
        df.to_excel(output_dir / filename, index=False)

        # error rate
        overall_preds = np.array(perturbed_pred, dtype=str)
        overall_labels = np.array(original_pred, dtype=str)
        n_total = len(overall_preds)
        err = sum(overall_preds != overall_labels) / n_total if n_total > 0 else None
        err_n2p = sum((overall_labels=="-1") & (overall_preds=="1")) / n_total if n_total > 0 else None
        err_p2n = sum((overall_labels=="1") & (overall_preds=="-1")) / n_total if n_total > 0 else None
        rer = err_n2p + err_p2n if err_n2p is not None and err_p2n is not None else None

        test_summary["test name"] = test_name
        test_summary["number of support"] = n_total
        test_summary["error rate"] = err
        test_summary["reversal error rate"] = rer
        test_summary["error-neg2pos"] = err_n2p
        test_summary["error-pos2neg"] = err_p2n
        test_summary["filename"] = filename

        return test_summary
