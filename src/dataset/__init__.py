from dataset.target_classification import TargetClassificationDataset
from dataset.chinese_word_segmentation import ChineseWordSegmentationDataset
from dataset.sequence_classification import SequenceClassificationDataset


TASK_TO_DATASET = {
    'target_classification': TargetClassificationDataset, 
    'chinese_word_segmentation': ChineseWordSegmentationDataset, 
    'sequence_classification': SequenceClassificationDataset, 
}


def get_dataset(dataset, tokenizer, args):
    return TASK_TO_DATASET[args.task](dataset=dataset, tokenizer=tokenizer, args=args)