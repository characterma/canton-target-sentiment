
# Installation

## Install requirements.txt
```bash
pip install -r requirements.txt
```

## For development
```bash
python -m pip install -e '.[dev]'
```

## For deployment
```bash
python -m pip install '.[deploy]'
```

# Train
Go to nlp_pipeline/ directory.

For task X and model Y, 
```bash
python run.py --config_dir="../config/examples/X/Y"
```

# Test
Go to nlp_pipeline/ directory.

For task X and model Y, 
```bash
python run.py --config_dir="../config/examples/X/Y" --test_only
```

# Explanation (only for `sequence_classification`)
Go to nlp_pipeline/ directory.

```bash
python run.py --config_dir="../config/examples/sequence_classification/BERT_AVG_explain" --test_only --explain --faithfulness
```