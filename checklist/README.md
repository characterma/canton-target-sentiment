# Checklist

## Installation
```bash
pip install -r requirements.txt
pip install -r ailabuap.txt
```

## Configuration
All configurations should be set within `checklist.yaml` file. Please refer to the config example under `/config/checklist/<task_name>/checklist.yaml`. 

### 1. Model / API
Please select the suitable model type and specify the corresponding model path
- test with well-trained model via nlp_pipeline

```yaml
model_type: <pytorch/onnx/jit>
model_path: <the directory of trained model>
```

- test with API

```yaml
model_type: <target-senti-api/doc-senti-api>
model_path: <api endpoint>
```

### 2. Output directory
Please spectify the output directory `output_dir` to save checklist report.

### 3. Data resources
Define the data required in testcases, the key of the resource is used as an input argument in testcase.

- data directory

Please put all the data files under one directory and set as `data_dir`

- define resources

As shown in examples, resources can be defined in the format of {key: value}, key is the shortcut of resource and value is the data path with relative to `data_dir`. If the data key has two levels, for example entities contains different types, e.g. company, person etc. The shortcut should be {parent_key.children_key}.

- data format

Only support `.txt` and `.json` file. If the resource is a list of string, please construct it into a txt file. Otherwise, please format it as a json file with orient="records".

In terms of testing target sentiment model, only support `.json` file since the keyword_index is a must for inputs.

### 4. Test cases
Please define your testcases in this section. When excutes checklist, the checklist will go through all the testcases in this section and generate reports, saving in `output_dir`.

#### Minimum Functionality Test (MFT)

MFT is a test with the provided labels to check if the model can predict the correct label under the case.

- Case 1: To test **one** list of **words/sentence** with same label (not suitable for target sentiment)
For example, if we would like to test a set of positive words, we can define the test case as follow.

```yaml
- name: entity-company # the name of the test, this will show on the report as the identify of this case

  # capacity is to state the functionality of the test in case you would like to group the similar test case
  capacity: model bias on entity
  
  # operator is an built-in function for this type of testcase, we suggest to use test_with_template if you would like to test a list of strings without any other data files
  operator: test_with_template
  
  # template allows you to define the string format you want. You can add some prefix or suffix words before or after {key}, and {key} will be replaced by test strings. 
  template: "{input_a}"
  
  # specify the data resources to be used as inputs
  inputs:
    input_a: entity.company # resources shortcut or data path
  
  # the label for this set of data
  label: 0
  
  # nsamples is optional, if not set, all the data points will be tested
  nsamples: 10
```

- Case 2: To test **two** list of **words/sentence** with same label (not suitable for target sentiment)
For example, if you would like to test a set of positive words with a combination of entity, we can difine the test case as follow.

```yaml
- name: positive-company
  capacity: entity + sentiment word
  operator: test_with_template # similar to case 1, we use test_with_template as operator 
  template: "{input_a}{input_b}" # use two different keys for different data resources 
  inputs:
    input_a: entity.company
    input_b: positive
  label: 1
  nsamples: 10
```

- Case 3: To test one list of **data dict** with same label

```yaml
- name: positive
  capacity: model generalization on positive words
  # To use original data dict as inputs, we suggest to use test_with_raw_inputs as operator. This operator will directly forward each data dict to model/API without changing any data_fields.
  operator: test_with_raw_inputs
  inputs:
    input_a: positive
  label: 1
  nsamples: 10
```

- Case 4: To test one list of **data dict** with one corresponding label for each data record

```yaml
- name: testset-data
  capacity: evaluate test accuracy
  # To use original data dict as inputs, we suggest to use test_with_raw_inputs as operator. This operator will directly forward each data dict to model/API without changing any data_fields.
  operator: test_with_raw_inputs
  inputs:
    input_a: labelled_data
  # Notice: to use the original labels, please do NOT set label or keep it as None
  label: None
  nsamples: 10
```

#### Invariance tests (INV)
An Invariance test (INV) is when we apply label-preserving perturbations to inputs and expect the model prediction to remain the same.
Currently we support the following pururbations for both doc-level and target-level:

  - Random deletion
  - Keyword deletion
  - Entity deletion
  - Keyword replacement
  - Entity replacement
  - Add typo (implemented by keyword replacement)
  - Remove punctuation (implemented by keyword deletion)

For the usage of each pertubation, we provide an example as follow:

**1 - Random deletion**

Randomly remove part of the words/characters, which are not adj., adv., v., negation word.

```yaml
- name: random-deletion
  capacity: robustness
  operator: remove_random # function name for the operation
  inputs:
    data: labelled_data # can use labelled or unlabelled data
    segmentation: word # word/char
    purturbation_ratio: 0.15 # the ratio of segment to be removed
    samples: 1 # number of samples generated by each original sample
  nsamples: 10
```

**2 - Keyword deletion**

Remove part of the provided target words.

```yaml
- name: keyword-deletion-product
  capacity: robustness
  operator: remove_keyword # function name for the operation
  inputs:
    data: labelled_data # can use labelled or unlabelled data
    keywords: entity.product
    purturbation_ratio: 0.15
    samples: 1
  nsamples: 10
```

**3 - Entity deletion**

Apply NER (API endpoint) on each sentence and remove one entity belonging to the selected entity type if exists.

```yaml
- name: entity-deletion-company
  capacity: robustness
  operator: remove_entity
  inputs:
    data: labelled_data # can use labelled or unlabelled data
    NER_target_field: company # in this example, if there are company entities, one of them will be removed
    NER_endpoint: http://ess25.wisers.com/playground/ner-kd-jigou-gpu/entityextr/analyse
    samples: 1
  nsamples: 10
```

**4 - Keyword replacement**

```yaml
- name: keyword-replacement-company
  capacity: robustness
  operator: change_keyword
  inputs:
    data: labelled_data # can use labelled or unlabelled data
    keywords: entity.company
    purturbation_ratio: 0.2
    samples: 1
  nsamples: 10
```

**5 - Entity replacement**

```yaml
- name: entity-replacement-company
  capacity: robustness
  operator: change_entity
  inputs:
    data: labelled_data # can use labelled or unlabelled data
    NER_target_field: company # the target filed name extracted by NER
    NER_endpoint: http://ess25.wisers.com/playground/ner-kd-jigou-gpu/entityextr/analyse
    alternatives: entity.company # a list of alternatives to replace
    samples: 3
  nsamples: 10
```

**6 - Add typo (implemented by keyword replacement)**

```yaml
- name: add_typo
  capacity: denoising
  operator: change_dict
  inputs:
    data: labelled_data
    replacement_dict: typo
    purturbation_ratio: 0.2
    samples: 1
```

**7 - Remove punctuation (implemented by keyword deletion)**

```yaml
- name: strip_punctuation # need an argument to set sub char?
  capacity: denoising
  operator: remove_keyword
  inputs:
    data: labelled_data
    keywords: entity.punctuation
    purturbation_ratio: 0.2
  nsamples: 10
```

## Execute checklist

```bash
cd sentiment/
python run_checklist.py --config_dir <the directory contains checklist.yaml>
```

### Examples
``` bash
cd sentiment/

# Document sentiment model
python run_checklist.py --config_dir ../../config/doc_senti_model/

# Document sentiment API
python run_checklist.py --config_dir ../../config/doc_senti_api/

# Target sentiment model
python run_checklist.py --config_dir ../../config/target_senti_model/

# Target sentiment API
python run_checklist.py --config_dir ../../config/target_senti_api/
```
