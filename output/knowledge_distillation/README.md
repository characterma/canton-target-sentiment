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
|                                                 | On train set |          | On test set |          |
|-------------------------------------------------|--------------|----------|-------------|----------|
|                                                 | Micro-F1     | Macro-F1 | Micro-F1    | Macro-F1 |
| Teacher model (BERT_CLS)                        | 84.81%       | 81.48%   | 86.41%      | 84.12%   |
| Student model (TEXT_CNN, tokenizer: internal)   | 84.32%       | 81.07%   | 82.54%      | 79.67%   |
| Student model (TEXT_CNN, tokenizer: char_split) | 82.28%       | 78.59%   | 83.01%      | 80.15%   |