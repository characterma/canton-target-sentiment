# Installation

## Install requirements.txt
```bash
pip install -r requirements.txt
```

## For development
```bash
python -m pip install -e '.[dev]'
```

# Unittest

```bash
python -m unittest
```

## For deployment
```bash
python -m pip install '.[deploy]'
```

# Train
Go to nlp_pipeline/ directory.

For task X and model Y, 
```bash
python run.py --config_dir="../config/examples/X/Y"
```

# Test
Go to nlp_pipeline/ directory.

For task X and model Y, 
```bash
python run.py --config_dir="../config/examples/X/Y" --test_only
```

# Explanation (only for `sequence_classification`)
Go to nlp_pipeline/ directory.

```bash
python run.py --config_dir="../config/examples/sequence_classification/BERT_AVG_explain" --test_only --explain --faithfulness
```

# Build optimization models

Go to nlp_pipeline/ directory.

For task X and model Y, 

## Onnx

export model to onnx
```bash
python build_onnx.py --config_dir='../config/examples/X/Y'
```

conduct onnx model quantization
```bash
python optimize_onnx.py --config_dir='../config/examples/X/Y'
```

## Jit trade

```bash
python build_jit_trace.py --config_dir='../config/examples/X/Y'
```

# MLOps logging 
Train a sequence_classification model with experiment tracking on tools
```bash
python run.py --config_dir="../config/examples/sequence_classification/BERT_AVG_mlops" 
```

adjust mlops config by edition of run.yaml file
- login neptune dashboard: wisemen930@gmail.com

```yaml
...
mlops:
  neptune:
    project: wisemen/spam-classification # create your own project
    api_token: # get API token of the project on neptune page
    name:  # default uses the folder name of experiment
    description: # describe the purpose of experiment
    mode: "async" # 'async' or 'debug' or 'offline' or 'read-only' or 'sync'. Ref [here](https://docs.neptune.ai/api/connection_modes/)
    log: False # log dev metrics during training or not (save logging time if False)
    tags: # list of tags for searching and comparing experiments 
    - 'bert'
    - 'spam'
    capture_hardware_metrics: True # default True, log the hardware usage
...
```

# KD Training 
Train KD model by selecting a config folder
```bash
python run.py --config_dir="../config/examples/sequence_classification/TEXT_CNN_kd" 
```

adjust KD config by edition of run.yaml file
- loss type 
  - definition: loss function
  - mse: KL temperature is not applied
  - kl: KL temperature finetuning may be reuired, the finetuned result can show better performnce than that of mse
- soft lambda 
  - definition : total loss = (1-lambda) * hardLableLoss + lambda * softLableLoss (KD loss), lambda between 0 to 1
  - finetuned first choice 0.1, 0.5, 1.0
  - finetuned second choice 0.3, 0.7
- KL temperature (only for loss type kl)
  - definition: positive scale scale to soften logit, softmax's scale: 1
  - finetuned choice: 1, 3, 5


```yaml
...
train:
  model_class: "TEXT_CNN"
  kd:
    use_kd: True
    teacher_dir: "../config/examples/sequence_classification/BERT_AVG/model/"
    loss_type: 'mse'
    soft_lambda: 0.5
    kl_T: 5
...
```


## Pretrained Embedding

From previous KD experiments, student TEXT CNN model shows better accruacy if the training procedure adopts:
1. transformer tokenizer 

- tokenizer_name
  - choices: pretrained tokenier from HuggingFace

```yaml
...
model_params:
  ...
  tokenizer_source: "transformers"
  tokenizer_name: "bert-base-chinese" # Huggingface tokenier
  ...
...
```

2. pretrained embedding 
    - pretrained_emb_path (default is random embedding if None)
      - format: text file, take reference from sample embedding "data/word_embeddings/sample_word_emb.txt", first line is metadata (number of vocab, number of dimension)
      - source: embedding file is not provided in directory, you can create one by executing [dim_reduction.py](nlp_pipeline/dim_reduction.py). The document is in [here](README.md/# KD Training Guildline/## Pretrained Embedding)
    - emb_dim: default is 80, please specify the dimension of pretrained embedding
    - remarks: select pretrained embedding which have same dictionary (same vocabuluries and order) as tokenier's vocab

```yaml
...
model_params:
  ...
  pretrained_emb_path: "../data/word_embeddings/bert_base_chinese_emb_64d.txt"
  emb_dim: 64
  ...
...
```

## Pretrained Embedding
Reference: https://jira.wisers.com:18090/pages/viewpage.action?spaceKey=RES&title=Proposed+Module

This program is used to reduce dimension and save of HuggingFace pretrained embedding.

- pretrain_path (either one): 
  1. local directory of text embedding file (include file name)
  2. Huggingface model
- output_dim
  - positive integer that smaller than originl dimension
- save_path
  - local directory that save result text file (include file name)
- reduction_mode
  - choice: 'PPA-PCA' (preferred), 'PCA-PPA', 'PCA', 'PPA-PCA-PPA'

Go to nlp_pipeline/ directory.
```bash
python dim_reduction.py \
--pretrain_path 'bert-base-chinese' \
--output_dim 64 \
--save_path '../data/word_embeddings/bert_base_chinese_64d.txt' \
--reduction_mode: 'PPA-PCA'
```

## Dynamic Temperature Distillation
Reference: https://jira.wisers.com:18090/pages/viewpage.action?spaceKey=RES&title=Proposed+Module

This module is effective to enhance performance of KD training in small set (data size < 3k). It requires to train using KL loss function and hyper parameter finetuning on DTD configuration.

- dtd_type
  - choice: flsw (preferred), cwsm
- dtd bias 
  - finetuned choice: (10, 20, 40, 80)
- dtd flsw power (only applicable on flsw type)
  - finetuned choice: (0.25, 0.5, 1)  

```yaml
...
train:
  model_class: "TEXT_CNN"
  kd:
    use_kd: True
    teacher_dir: "../config/examples/sequence_classification/BERT_AVG/model/"
    loss_type: 'kl'
    soft_lambda: 0.1
    kl_T: 1
    dtd_type: "flsw"
    dtd_bias: 20
    dtd_flsw_pow: 0.5
...
```

## Unlabel Data Augmentation

```yaml
...
data:
  ...
  unlabled:         # it indicates no data augmentation part in this training if no value in the key 
...
```

In order to learn from high quality unlabeled data, [unlabel_data_sampling notebook](notebook/unlabel_data_sampling.ipynb) can be used to filter and save those data. Two files will be generated:
- sampled_content.json (unlabel data for data augmentation)
- sampled_logits.pkl (skip the inference process of unlabel data in training time,following below steps:
1. rename the pickle file as 'logits_unlabeled.pkl'
2. copy it to experiment model directory (eg. TEXT_CNN_kd/model/)
)

Reference: https://jira.wisers.com:18090/display/RES/Proposed+Module2

## Active Learning
```yaml
...
data:
  al_unlabel: al_sample.json  # File name of the unlabel data set
  additional_train_dir: "../data/datasets/sample/spam_classification/active_learning" # Dir for the folder that stores additional training data

active_learning:
  use_al: True # use active learning to query data to label
  output_dir: "../data/datasets/sample/spam_classification/active_learning" # Dir for the folder that stores query active learning data
  output_file: "active_learning_queried_data" # File name for the query active learning data, a number will be added to the end automaticaly, e.g. "active_learning_queried_data_0.json"
  query_method: "prediction_entropy" # Method for quering data, see below
  query_size: 10 # Number of data to query
  # Below setting can be removed, it is for running experiment only
  run_al_exp: False # Please set it to False if you are not trying to run active learning iteration experiment
  result_file: "iteration_results.pkl" # File name for the training result of each iteration, will be stored in ../active_learning_result folder
  iteration: 10 # Number of active learning result to run 
...
```
Reference: https://jira.wisers.com:18090/display/WAI/Active+Learning+on+Text+classification

### Implemented query_method:
- Prediction Entropy: "prediction_entropy"
- Least Confidence: "least_confidence"
- Breaking Ties: "breaking_ties"
- Contrastive Active Learning(CAL): "cal" (It is required to use <cls> embedding, therefore the training model need to contain a BERT pre-train model)
- Coreset: "coreset" (It is required to use <cls> embedding, therefore the training model need to contain a BERT pre-train model)
- Random Sample: "random_sample"


### Note for running active learning:
- You can follow configs/examples/sequence_classification/BERT_AVG_AL/run.yaml as an example
- train, dev, test, al set SHOULD all contain "docid", it is to prevent querying duplicated data, errors may occur if it is absent
- Contrastive Active Learning will take a longer time if the al set is too large, it is recommended to NOT put ALL the unlabel data together to query by active learning