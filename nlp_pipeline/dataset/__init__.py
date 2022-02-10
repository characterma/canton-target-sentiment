from nlp_pipeline.dataset.classification import *
from nlp_pipeline.dataset.tagging import *
from nlp_pipeline.dataset.clue import *
from nlp_pipeline.dataset.base import NLPDataset


TASK_TO_FEATURE = {
    "target_classification": TargetClassificationFeature,
    "chinese_word_segmentation": ChineseWordSegmentationFeature,
    "sequence_classification": SequenceClassificationFeature,
}


def get_dataset(dataset, tokenizer, args, raw_data=None):

    return NLPDataset(
        feature_class=get_feature_class(args),
        dataset=dataset,
        tokenizer=tokenizer,
        args=args,
        raw_data=raw_data
    )


def get_feature_class(args):
    return TASK_TO_FEATURE[args.task]
