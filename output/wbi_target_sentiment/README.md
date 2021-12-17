# Instruction

## 1. Copying datasets
First, set current directory=src/. Then

```bash
mkdir -p ../data/datasets/internal/target_sentiment/wbi_comb
cp -r /ailab/shared/Users/tonychan/wbi_handover/data/wbi_comb/* ../data/datasets/internal/target_sentiment/wbi_comb/
```

## 2. Model training
For experiment xxx (set current directory=src/), 
```bash
python run.py --config_dir="../output/wbi_target_sentiment/org_per_bert_avg_20210925_all_ext2"
```

## 3. Model testing
For experiment xxx (set current directory=src/),
```bash
python run.py --config_dir="../output/wbi_target_sentiment/org_per_bert_avg_20210925_all_ext2" --test_only
```