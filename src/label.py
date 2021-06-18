import json
import os
from preprocess import TextPreprocessor
from dataset.chinese_word_segmentation import get_token_level_tags


def get_label_to_id(tokenizer, args):
    label_to_id_path = args.model_dir / "label_to_id.json"

    if os.path.exists(label_to_id_path):
        label_to_id = json.load(open(label_to_id_path, 'r'))
    else:
        label_to_id = load_label_to_id_from_datasets(
            datasets=['train'], 
            tokenizer=tokenizer, 
            args=args
        )
        json.dump(label_to_id, open(label_to_id_path, 'w'))

    label_to_id_inv = dict(zip(label_to_id.values(), label_to_id.keys()))
    return label_to_id, label_to_id_inv


def load_label_to_id_from_datasets(datasets, tokenizer, args):

    if args.task=='target_classification':
        labels = args.run_config['data']['labels']
        if labels=="2_ways":
            label_to_id = {"neutral": 0, "non_neutral": 1}
        elif labels=="3_ways":
            label_to_id = {"neutral": 0, "negative": 1, "positive": 2}
        else:
            raise ValueError("Label type not supported.")

    elif args.task=='chinese_word_segmentation':
        label_to_id = {'X': 0}
        tags = []

        # Scan all data and get tags
        files = []
        for dataset in datasets:
            files.append(args.data_config[dataset])
        files = list(set(files))
        
        for filename in files:
            data_path = args.data_dir / filename
            raw_data = json.load(open(data_path, 'r'))

            # text preprocessing
            for data_dict in raw_data:
                preprocessor = TextPreprocessor(
                    text=data_dict['content'], 
                    steps=args.prepro_config['steps']
                )
                content = preprocessor.preprocessed_text

                # tokenization
                tokens_encoded = tokenizer(
                    content, 
                    max_length=args.model_config['max_length'], 
                    add_special_tokens=False, 
                    return_offsets_mapping=False,
                    return_length=True,
                )

                # token level tags
                tags.extend(get_token_level_tags(
                    tokens_encoded, 
                    data_dict['sent_indexs'], 
                    data_dict['postags'], 
                    scheme='BI'
                ))

        for t in set(tags):
            label_to_id[t] = len(label_to_id)
    else:
        raise ValueError("Task not supported.")

    return label_to_id
