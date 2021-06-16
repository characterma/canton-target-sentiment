from dataset.target_classification import TargetClassificationDataset
from dataset.chinese_word_segmentation import ChineseWordSegmentationDataset


TASK_TO_DATASET = {
    'target_classification': TargetClassificationDataset, 
    'chinese_word_segmentation': ChineseWordSegmentationDataset
}


def get_dataset(dataset, tokenizer, args):
    task = args.run_config['train']['task']
    return TASK_TO_DATASET[task](dataset=dataset, tokenizer=tokenizer, args=args)