import sklearn
from seqeval.scheme import IOB2
from seqeval.metrics import classification_report as seqeval_classification_report


def compute_metrics(task, labels, predictions):
    if task in ["target_classification", "sequence_classification"]:
        report = compute_metrics_sequence_classification(labels, predictions)
    elif task == "chinese_word_segmentation":
        report = compute_metrics_sequence_tagging(labels, predictions)
    return report


def compute_metrics_sequence_classification(labels, predictions):

    report = sklearn.metrics.classification_report(
        labels, predictions, output_dict=True
    )
    labels_unique = set(labels)
    metrics = {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "micro_f1": report["weighted avg"]["f1-score"],
        "support": report["macro avg"]["support"],
    }
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
