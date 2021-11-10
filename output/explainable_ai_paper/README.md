# Instruction

## 1. Copying datasets
* For ag experiments (set current directory=src/):

```bash
mkdir -p ../data/datasets/public/sequence_classification/ag
cp -r /ailab/shared/Users/tonychan/xai_handover/data/sequence_classification/ag/* ../data/datasets/public/sequence_classification/ag
```

* For imdb experiments (set current directory=src/):

```bash
mkdir -p ../data/datasets/public/sequence_classification/imdb
cp -r /ailab/shared/Users/tonychan/xai_handover/data/sequence_classification/imdb/* ../data/datasets/public/sequence_classification/imdb
```

* For sst experiments (set current directory=src/):

```bash
mkdir -p ../data/datasets/public/sequence_classification/sst
cp -r /ailab/shared/Users/tonychan/xai_handover/data/sequence_classification/sst/* ../data/datasets/public/sequence_classification/sst
```

## 2. Model training
For experiment `X` (set current directory=src/), 
```bash
python run.py --config_dir="../output/explainable_ai_paper/X/"
```

## 3. Generating interpretations
For experiment `X` (set current directory=src/),
```bash
python run.py --config_dir="../output/explainable_ai_paper/X/" --test_only --explain --faithfulness
```