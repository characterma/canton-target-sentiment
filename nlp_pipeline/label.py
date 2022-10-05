import json
import os

from nlp_pipeline.preprocess import Preprocessor
from nlp_pipeline.dataset.tagging.utils import get_token_level_tags


def get_label_to_id(tokenizer, args, raw_data=None):
    print(os.listdir(args.model_dir))
    label_to_id_path = args.model_dir / "label_to_id.json"

    if os.path.exists(label_to_id_path):
        label_to_id = json.load(open(label_to_id_path, "r"))
    else:
        label_to_id = load_label_to_id_from_datasets(
            datasets=["train"], tokenizer=tokenizer, args=args, raw_data=raw_data
        )
        json.dump(label_to_id, open(label_to_id_path, "w"))

    label_to_id_inv = dict(zip(label_to_id.values(), label_to_id.keys()))
    return label_to_id, label_to_id_inv


def load_label_to_id_from_datasets(datasets, tokenizer, args, raw_data=None):

    if args.task == "target_classification":
        label_to_id = {}
        files = []
        for dataset in datasets:
            files.append(args.data_config[dataset])
        files = list(set(files))
        for filename in files:
            data_path = args.data_dir / filename
            if raw_data is None:
                raw_data = json.load(open(data_path, "r"))
            # text preprocessing
            for data_dict in raw_data:
                label = str(data_dict["label"])
                if label not in label_to_id:
                    label_to_id[label] = len(label_to_id)
        return label_to_id

    elif args.task == "chinese_word_segmentation":
        label_to_id = {"X": 0}
        tags = []

        # Scan all data and get tags
        files = []
        for dataset in datasets:
            files.append(args.data_config[dataset])
        files = list(set(files))

        for filename in files:
            data_path = args.data_dir / filename
            if raw_data is None:
                raw_data = json.load(open(data_path, "r"))

            # text preprocessing
            for data_dict in raw_data:
                preprocessor = Preprocessor(
                    data_dict=data_dict, steps=args.prepro_config["steps"]
                )
                content = preprocessor.data_dict["content"]

                # tokenization
                tokens_encoded = tokenizer(
                    content,
                    max_length=args.model_config["max_length"],
                    add_special_tokens=False,
                    padding="max_length",
                    return_offsets_mapping=False,
                    return_length=True,
                )
                # token level tags
                tags.extend(
                    get_token_level_tags(
                        tokens_encoded,
                        data_dict["sent_indexs"],
                        data_dict["postags"],
                        scheme="BI",
                    )
                )
        for t in set(tags):
            label_to_id[t] = len(label_to_id)
    elif args.task == "sequence_classification":
        label_to_id = {}
        files = []
        for dataset in datasets:
            files.append(args.data_config[dataset])
        files = list(set(files))
        for filename in files:
            data_path = args.data_dir / filename
            if raw_data is None:
                raw_data = json.load(open(data_path, "r"))
            # text preprocessing
            for data_dict in raw_data:
                label = str(data_dict["label"])
                if label not in label_to_id:
                    label_to_id[label] = len(label_to_id)
        return label_to_id
    elif args.task == "sequence_labeling":
        label_to_id = {}
        label_list = []
        files = []
        for dataset in datasets:
            files.append(args.data_config[dataset])
        files = list(set(files))
        for filename in files:
            data_path = args.data_dir / filename
            if raw_data is None:
                raw_data = json.load(open(data_path, "r"))
            # text preprocessing
            for data_dict in raw_data:
                labels = list(set(list(data_dict["labels"])))
                for label in labels:
                    if label not in label_list:
                        label_list.append(label)

        label_list.sort()
        label_to_id = {label: i for i, label in enumerate(label_list)}

        print(label_to_id)
        return label_to_id
    else:
        raise ValueError("Task not supported.")
    return label_to_id
