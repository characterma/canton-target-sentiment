# Instruction

## 1. Copying datasets & pretrained word embeddings
* For dataset `X` (set current directory=src/):

```bash
mkdir -p ../data/datasets/public/sequence_classification/X
cp -r /ailab/shared/Users/tonychan/xai_handover/data/sequence_classification/X/* ../data/datasets/public/sequence_classification/X
```

* For `text_cnn` models (set current directory=src/):

```bash
mkdir -p ../data/word_embeddings/
cp /ailab/shared/Users/tonychan/xai_handover/data/glove_840B_300d_vectors.txt ../data/word_embeddings/
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