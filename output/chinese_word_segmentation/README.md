# Instruction

## 1. Copying datasets
* For canton experiment (first set current directory as src/)

```bash
mkdir -p ../data/datasets/internal/cws/canton
cp -r /ailab/shared/Users/tonychan/cws_handover/data/canton/* ../data/datasets/internal/cws/canton/
```

* For ctb5 experiment (first set current directory as src/)

```bash
mkdir -p ../data/datasets/public/cws/ctb5
cp -r /ailab/shared/Users/tonychan/cws_handover/data/ctb5/* ../data/datasets/public/cws/ctb5/
```


## 2. Model training
For experiment `X` (first set current directory=src/), 
```bash
python run.py --config_dir="../output/chinese_word_segmentation/X/"
```

## 3. Model testing
For experiment `X` (first set current directory=src/),
```bash
python run.py --config_dir="../output/chinese_word_segmentation/X/" --test_only
```