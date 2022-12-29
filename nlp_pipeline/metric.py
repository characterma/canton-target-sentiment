from seqeval.scheme import IOB2
from seqeval.metrics import classification_report as seqeval_classification_report
from sklearn.metrics import classification_report as sklearn_classification_report
from sklearn.preprocessing import MultiLabelBinarizer

def compute_metrics(args, label_ids, predictions, prediction_probas=None):
    labels = []

    if args.task in ["target_classification", "sequence_classification"]:
        for l1, p1 in zip(label_ids, predictions):
            labels.append(args.label_to_id_inv[l1])
        report = compute_metrics_sequence_classification(labels, predictions, prediction_probas=prediction_probas, label_to_id=args.label_to_id)
    elif args.task == "chinese_word_segmentation":
        for l1, p1 in zip(label_ids, predictions):
            labels.append(list(map(lambda x: args.label_to_id_inv[x], l1))[: len(p1)])
        report = compute_metrics_sequence_tagging(labels, predictions)
    elif args.task == 'topic_classification':
        import numpy as np
        for l1, p1 in zip(label_ids, predictions):
            labels.append([args.label_to_id_inv[i] for i,l in enumerate(l1) if l == 1])
        report = compute_metrics_topic_classification(labels, predictions, args.label_to_id)
    return report


def compute_metrics_sequence_classification(labels, predictions, prediction_probas=None, label_to_id=None):

    report = sklearn_classification_report(
        labels, predictions, output_dict=True
    )
    labels_unique = set(labels)
    metrics = {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "micro_f1": report["weighted avg"]["f1-score"],
        "support": report["macro avg"]["support"],
    }
    if prediction_probas and label_to_id:
        # Calculate Area under Precision-Recall curve
        import pandas as pd
        from sklearn.metrics import precision_recall_curve, auc

        df = pd.DataFrame({"labels": labels, "predictions": prediction_probas})
        for label in labels_unique:
            probas_id = label_to_id[label]
            precision, recall, _ = precision_recall_curve(df["labels"] == label, [probas[probas_id] for probas in df["predictions"]])
            auc_score = auc(recall, precision)
            metrics[f"{label}-auc"] = auc_score

    for label, v1 in report.items():
        if label in labels_unique:
            for score_name, v2 in v1.items():
                metrics[f"{label}-{score_name}"] = v2
    return metrics


def compute_metrics_sequence_tagging(labels, predictions):
    report = seqeval_classification_report(
        labels, predictions, output_dict=True, scheme=IOB2
    )

    metrics = {
        # "acc": report['accuracy'],
        "macro_f1": report["macro avg"]["f1-score"],
        "micro_f1": report["weighted avg"]["f1-score"],
        "support": report["macro avg"]["support"],
    }
    for label, v1 in report.items():
        if label not in ["micro avg", "macro avg", "weighted avg"]:
            for score_name, v2 in v1.items():
                metrics[f"{label}-{score_name}"] = v2
    return metrics

def compute_metrics_topic_classification(labels, predictions, label_to_id):

    sort_label_to_id = dict(sorted(label_to_id.items(), key=lambda item: item[1]))
    label_type = list(sort_label_to_id.keys())
    mlb = MultiLabelBinarizer(classes = label_type)
    labels_mat = mlb.fit_transform(labels)
    preds_mat = mlb.fit_transform(predictions)
    report = sklearn_classification_report(
        labels_mat, preds_mat, target_names=label_type, output_dict=True
    )

    metrics = {
        "macro_f1": report["macro avg"]["f1-score"],
        "micro_f1": report["weighted avg"]["f1-score"],
        "support": report["macro avg"]["support"]
    }
    for label, v1 in report.items():
        if label in label_type:
            for score_name, v2 in v1.items():
                metrics[f"{label}-{score_name}"] = v2
    return metrics