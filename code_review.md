## Code reusability

### Scope:
1. Target sentiment projects
1. NLP projects

### Reusability per module

* preprocess.py
  * TextPreprocessor: 2 (new methods might be required)
* dataset.py
  * pad_tensor: 2
  * load_vocab: 2
  * build_vocab_from_pretrained: 2
  * build_vocab_from_dataset: 2
  * TargetDependentExample: 1 (modification might be required)
  * TargetDependentDataset: 1 (modification might be required)
* tokenizer.py
  * get_tokenizer: 2
  * TokensEncoded: 2
  * InternalTokenizer: 2
* trainer.py
  * compute_metrics: 2
  * prediction_step: 1
  * evaluate: 1
  * Trainer: 2
* utils.py
  * set_seed: 2
  * load_yaml: 2
  * set_log_path: 2
  * get_label_to_id: 1
  * load_config: 2
* run.py
  * init_model: 2
  * init_tokenizer: 2
  * run: 2
  * combine_and_save_metrics: 2
  * combine_and_save_statistics: 2
* model/__init__.py
  * get_model: 2
  * get_model_type: 2
* model/model_utils.py
  * load_pretrained_bert: 2
  * load_pretrained_config: 2
  * load_pretrained_emb: 2
  * BaseModel: 2
* TDBERT.py
  * TDBERT: 1
* TGSAN.py
  * TGSAN: 1
* TGSAN2.py
  * TGSAN2: 1

### Reusability summary
* Number of classes / functions: 33
* Reusable by other target sentiment projects: 33 / 33 = 100%
* Reusable by other NLP projects: 25 / 33 = 75.76%