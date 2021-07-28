from dataset.target_classification import TargetClassificationFeature
from dataset.chinese_word_segmentation import ChineseWordSegmentationFeature
from dataset.sequence_classification import SequenceClassificationFeature
from dataset.base import NLPDataset


TASK_TO_FEATURE = {
    "target_classification": TargetClassificationFeature,
    "chinese_word_segmentation": ChineseWordSegmentationFeature,
    "sequence_classification": SequenceClassificationFeature,
}


def get_dataset(dataset, tokenizer, args):

    return NLPDataset(
        feature_class=get_feature_class(args),
        dataset=dataset,
        tokenizer=tokenizer,
        args=args,
    )


def get_feature_class(args):
    return TASK_TO_FEATURE[args.task]
