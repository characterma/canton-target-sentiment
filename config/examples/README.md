# Introduction

This tutorial covers 
- Data preparation for the nlp_pipeline.
- How to use the nlp_pipeline for model training & testing.
- What is included in the output files

# Data preparation

Supported tasks:
* sequence_classification
* target_classification
* chinese_word_segmentation

Each task has a standard data format. You need to find the data format by
1. Know your task. It should be one of the above.
1. Go to `canton-target-sentiment/data/datasets/sample/`
1. Go into the sub-directory with the same task name.
1. Open `sample.json`

`sample.json` contains a list of dict. Each dict has several fields and is an instance of standardized data. You should transform you datasets into that format as json files.

# Training & testing

After you have transformed your datasets, you need to configurate the `run.yaml` file in order to run the pipeline. 

You can find example for `run.yaml` for each task and model:
1. Know your task and model.
1. Go to `canton-target-sentiment/config/example/`
1. Go into the sub-directory with the same task name.
1. Go into the sub-directory with the same model name.
1. Copy the content of `./run.yaml` to  `canton-target-sentiment/config/run.yaml`

The fields inside this `run.yaml` controls the task, data, preprocessing, model, and more. You should at least update the `data` field in which you define the `output_dir`, `data_dir`, and data files names.

# Train

Go into `canton-target-sentiment/nlp_pipeline/`.

```bash
python run.py
```

# Test (train first!)

After you train a model, you should know where the output folder is (defined in `run.yaml`).
Go to `canton-target-sentiment/nlp_pipeline/`.

```bash
python run.py --config_dir="{output_dir}/model" --test_only
```
where `{output_dir}` is substituted by actual value of `output_dir`.

# Explaining output files

After you train your model, inside `output_dir`, you will find `model/` and `result/`:

* `model/` contains the model file, tokenizer files, and config files for re-producing the results or deploying the model.
* `result/` 
  * `result.csv` which contains the evaluation metrics of your trained model on each dataset
  * `diagnosis.csv` which contains intermediate variables (including tokens, features) created for each data instance
  * `statistics.csv` which contains the statistics of your datasets