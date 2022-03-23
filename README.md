
# Installation

## Install requirements.txt
```bash
pip install -r requirements.txt
```

## For development
```bash
python -m pip install -e '.[dev]'
```

# Unittest

```bash
python -m unittest
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

# Build optimization models

Go to nlp_pipeline/ directory.

For task X and model Y, 

## Onnx

```bash
python build_onnx.py --config_dir='../config/examples/X/Y'
```

## Jit trade

```bash
python build_jit_trace.py --config_dir='../config/examples/X/Y'
```