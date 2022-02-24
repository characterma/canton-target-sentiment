# Examine (and update) config file

1. Go to config directory, for example, `./sequence_classification/BERT_CLS`, 
1. Open `run.yaml` with editor
1. Update parameters. (`task`, `model_class`, `data`, etc)

# Train particular task & model

Go to nlp_pipeline/ directory.

```bash
python run.py --config_dir="../config/examples/task/model" 
```

For example, for sequence classification with BERT_CLS,

```bash
python run.py --config_dir="../config/examples/sequence_classification/BERT_CLS" 
```

# Test particular task & model (train first!)

Go to nlp_pipeline/ directory.

```bash
python run.py --config_dir="../config/examples/task/model" --test_only
```

For example, for sequence classification with BERT_CLS,

```bash
python run.py --config_dir="../config/examples/sequence_classification/BERT_CLS" --test_only
```

