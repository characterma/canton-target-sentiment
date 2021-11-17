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
python run.py --config_dir="../output/knowledge_distillation/teacher_model_cls"
```

## 3. Student model training
First set current directory=src/, 
```bash
python run.py --config_dir="../output/knowledge_distillation/student_model"
```

# Results

|                          | Micro-F1 | Macro-F1 |
|--------------------------|----------|----------|
| Teacher model (BERT_CLS) | 86.41%   | 84.12%   |
| Student model (TEXT_CNN) | 82.54%   | 79.67%   |