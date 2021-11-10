# Instruction

# 1. Training
Go to src/ directory.

For task X and model Y, 
```bash
python run.py --config_dir="../config/examples/X/Y"
```

# 2a. Testing
Go to src/ directory.

For task X and model Y, 
```bash
python run.py --config_dir="../config/examples/X/Y" --test_only
```

# 2b. Explanation (only for `sequence_classification`)
Go to src/ directory.

```bash
python run.py --config_dir="../config/examples/sequence_classification/BERT_AVG_explain" --test_only --explain --faithfulness
```