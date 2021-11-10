# Instruction

## 1. Copying datasets
First set current directory as src/,

```bash
mkdir -p ../data/datasets/internal/sequence_classification/post_sentiment
cp -r /ailab/shared/Users/tonychan/kd_handover/data/post_sentiment/* ../data/datasets/internal/sequence_classification/post_sentiment/
```


## 2. Teacher model training
First set current directory=src/, 
```bash
python run.py --config_dir="../output/knowledge_distillation/teacher_model"
```

## 3. Student model training
First set current directory=src/, 
```bash
python run.py --config_dir="../output/knowledge_distillation/student_model"
```
