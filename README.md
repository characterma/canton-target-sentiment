# **Target-dependent sentiment analysis**

## API Specification

**URL: /target_sentiment**

**Type: POST**

**Description**

1. Process target-guided sentiment analysis.
2. Chinese model are currently supported. 

### Input & output formats (general)


**Input Data**

- `language` indicates the language ("chinese" or "english")  of input text.
- `text`: contents.
- `target`: the indices of the target.
- `format`: 'general'

**Sample Input**

```json
{
    "language": "chinese",
    "format": "general",
    "text": "蕭邦手錶一直是上層社會的寵愛之物，但長期性的應用過程中在所難免出現一些常見故障，假如腕錶遭受強烈的撞擊，會給腕錶導致表針掉下來的狀況。",
    "target": [[0,2],[58,60],[58,60]],
}
```

**Output Data**

- Original input with an extra field:
  * sentiment ("0", "-1", or "1")

**Sample Output**

```json
{
    "language": "chinese",
    "format": "general",
    "text": "蕭邦手錶一直是上層社會的寵愛之物，但長期性的應用過程中在所難免出現一些常見故障，假如腕錶遭受強烈的撞擊，會給腕錶導致表針掉下來的狀況。",
    "target": [[0,2],[58,60],[58,60]],
    "sentiment": "-1",
}
```

### Input & output formats (syntactic API)

**Input Data**

- `language` indicates the language ("chinese" or "english")  of input text.
- `doclist`: contains one or multiple documents
- `labelunits`: contains labelunits in each document. Each labelunit has four required fields.
  * unit_index
  * unit_text
  * subject_index
  * aspect_index
- `format`: 'syntactic'

**Sample Input**

```json
{
  "language": "chinese",
  "format": "syntactic",
  "doclist": [
    {
      "labelunits": [
        {
          "unit_index": [0,67],
          "subject": "84ee5a1186004c528828cd24e96c7f6e",
          "unit_text": "蕭邦手錶一直是上層社會的寵愛之物，但長期性的應用過程中在所難免出現一些常見故障，假如腕錶遭受強烈的撞擊，會給腕錶導致表針掉下來的狀況。",
          "subject_name": "WATCH--CHOPARD",
          "subject_index": [[0, 2], [58, 60]], 
          "subject_text": ["蕭邦"],
          "subject_source": null,
          "aspect": "a02b98acb8434cb1a4b68ee4b9381cd7", 
          "aspect_name": "表针", 
          "aspect_index": [[58, 60]], 
          "aspect_text": ["表針"], 
          "sentiment": null, 
          "sentiment_index": [], 
          "sentiment_text": []
        }
        ]
    },
    {
      "labelunits": [
        {
            "unit_index": [0, 81], 
            "unit_text": "同樣道理，在黑暗中以紫外 燈照射roger dubuis excalibur blacklight所發出的七彩光芒，在剔透的鏤通機芯映襯下，也顯得更具深度及迷人。", 
            "subject": "66e23c65607e444ca3f844eb8156d4bb", 
            "subject_name": "ROGER DUBUIS--Excalibur王者系列", 
            "subject_index": [[16, 21], [22, 28], [29, 38], [39, 49]], 
            "subject_text": ["roger", "dubuis", "excalibur", "blacklight"], 
            "subject_source": null, 
            "aspect": "0e784175173349da824aa695148b484e", 
            "aspect_name": "机芯类型", 
            "aspect_index": [[64, 66]], 
            "aspect_text": ["機芯"], 
            "sentiment": null, 
            "sentiment_index": [], 
            "sentiment_text": []
        }
        ]
    }
  ]
}
```

**Output Data**

- Original input with an extra field for each labelunit:
  * sentiment ("0", "-1", or "1")

**Sample Output**

```json
{
    "language": "chinese",
    "format": "syntactic",
    "doclist": [
        {
            "labelunits": [
                {
                    "unit_index": [
                        0,
                        67
                    ],
                    "subject": "84ee5a1186004c528828cd24e96c7f6e",
                    "unit_text": "蕭邦手錶一直是上層社會的寵愛之物，但長期性的應用過程中在所難免出現一些常見故障，假如腕錶遭受強烈的撞擊，會給腕錶導致表針掉下來的狀況。",
                    "subject_name": "WATCH--CHOPARD",
                    "subject_index": [
                        [
                            0,
                            2
                        ],
                        [
                            58,
                            60
                        ]
                    ],
                    "subject_text": [
                        "蕭邦"
                    ],
                    "subject_source": null,
                    "aspect": "a02b98acb8434cb1a4b68ee4b9381cd7",
                    "aspect_name": "表针",
                    "aspect_index": [
                        [
                            58,
                            60
                        ]
                    ],
                    "aspect_text": [
                        "表針"
                    ],
                    "sentiment": "-1",
                    "sentiment_index": [],
                    "sentiment_text": []
                }
            ]
        },
        {
            "labelunits": [
                {
                    "unit_index": [
                        0,
                        81
                    ],
                    "unit_text": "同樣道理，在黑暗中以紫外 燈照射roger dubuis excalibur blacklight所發出的七彩光芒，在剔透的鏤通機芯映襯下，也顯得更具深度及迷人。",
                    "subject": "66e23c65607e444ca3f844eb8156d4bb",
                    "subject_name": "ROGER DUBUIS--Excalibur王者系列",
                    "subject_index": [
                        [
                            16,
                            21
                        ],
                        [
                            22,
                            28
                        ],
                        [
                            29,
                            38
                        ],
                        [
                            39,
                            49
                        ]
                    ],
                    "subject_text": [
                        "roger",
                        "dubuis",
                        "excalibur",
                        "blacklight"
                    ],
                    "subject_source": null,
                    "aspect": "0e784175173349da824aa695148b484e",
                    "aspect_name": "机芯类型",
                    "aspect_index": [
                        [
                            64,
                            66
                        ]
                    ],
                    "aspect_text": [
                        "機芯"
                    ],
                    "sentiment": "1",
                    "sentiment_index": [],
                    "sentiment_text": []
                }
            ]
        }
    ]
}
```

## API Load Test

**Description**
* Test case: 
  * Single document with single label unit.
  * Unit text length=81. (Median unit text length=72)

* K8S enviroment: 
  * cpu: 2000m
  * memory: 8G
  * maximum replicas: 1

**Results -- CPU only**

| Number of concurrent users | Number of requests processed | Requests per second | Failure | Maximum CPU utilization | Maximum memory utilization | GPU memory utilization |
|----------------------------|------------------------------|---------------------|---------|-------------------------|----------------------------|----------------------------|
| 20                         | 6323                         | 91.53               | 0       | 600m                    | 4.88G                      ||
| 50                         | 10169                        | 147.38              | 0       | 600m                    | 4.88G                      ||
| 100                        | 9988                         | 144.53              | 0       | 600m                    | 4.88G                      ||
| 200                        | 11458                        | 165.8               | 0       | 1200m                   | 5.90G                      ||

**Results -- with GPU (TGSAN)**

| Number of concurrent users | Number of requests processed | Requests per second | Failure | Maximum CPU utilization | Maximum memory utilization | GPU memory utilization |
|----------------------------|------------------------------|---------------------|---------|-------------------------|----------------------------|----------------------------|
| 20                         | 10094                         | 140.49               | 0       | <1000m                    | <1.20G                      |6.3GB|
| 50                         | 39730                        | 492.39              | 0       | <1000m                    | <1.20G                      |6.3GB|
| 100                        | 49828                         | 611.59              | 0       | <1000m                    | <1.20G                      |6.3GB|
| 200                        | 53229                        | 664.67               | 0       | <1000m                   | <1.20G                      |6.3GB|


**Results -- with GPU (TGSAN2)**
| Number of concurrent users | Number of requests processed | Requests per second | Failure | Maximum CPU utilization | Maximum memory utilization | GPU memory utilization |
|----------------------------|------------------------------|---------------------|---------|-------------------------|----------------------------|----------------------------|
| 20                         | 2585                         | 133.43               | 0       | <1000m                    | <1.20G                      |6.25GB|
| 50                         | 3425                        | 461.81              | 0       | <1000m                    | <1.20G                      |6.25GB|
| 100                        | 3499                         | 673.52              | 0       | <1000m                    | <1.20G                      |6.25GB|
| 200                        | 3534                        | 725.33               | 0       | <1000m                   | <1.20G                      |6.25GB|


**Results -- with GPU (TDBERT)**
| Number of concurrent users | Number of requests processed | Requests per second | Failure | Maximum CPU utilization | Maximum memory utilization | GPU memory utilization |
|----------------------------|------------------------------|---------------------|---------|-------------------------|----------------------------|----------------------------|
| 20                         | 2585                         | 36.97               | 0       | <1000m                    | <1.20G                      |7.75GB|
| 50                         | 3425                        | 46.47              | 0       | <1000m                    | <1.20G                      |7.75GB|
| 100                        | 3499                         | 50.12              | 0       | <1000m                    | <1.20G                      |7.75GB|
| 200                        | 3534                        | 50.53               | 0       | <1000m                   | <1.20G                      |7.75GB|


## Accuracy

**Description**
* Test data:
 * Up to 3000 samples from CN, HK, TW data.
 * Unseen during model training.

**Results (TGSAN)**

| Class    | Metric    | Score    |
|----------|-----------|----------|
| Overall  | Accuracy  | 0.8767 |
|          | Macro F1  | 0.7160  |
|          | Micro F1  | 0.8767 |
| Neutral  | Precision | 0.9206  |
|          | Recall    | 0.9345 |
|          | F1        | 0.9274 |
|          | Support   | 2519     |
| Negative | Precision | 0.6071 |
|          | Recall    | 0.5484 |
|          | F1        | 0.5763 |
|          | Support   | 62       |
| Positive | Precision | 0.6667 |
|          | Recall    | 0.6235 |
|          | F1        | 0.6444 |
|          | Support   | 494      |

**Results (TGSAN2)**

| Class    | Metric    | Score    |
|----------|-----------|----------|
| Overall  | Accuracy  | 0.8849 |
|          | Macro F1  | 0.7118  |
|          | Micro F1  | 0.8849 |
| Neutral  | Precision | 0.9136  |
|          | Recall    | 0.9571 |
|          | F1        | 0.9349 |
|          | Support   | 2519     |
| Negative | Precision | 0.5538 |
|          | Recall    | 0.5806 |
|          | F1        | 0.5669 |
|          | Support   | 62       |
| Positive | Precision | 0.7385 |
|          | Recall    | 0.5547 |
|          | F1        | 0.6335 |
|          | Support   | 494      |

**Results (TDBERT)**

| Class    | Metric    | Score    |
|----------|-----------|----------|
| Overall  | Accuracy  | 0.9273 |
|          | Macro F1  | 0.8192  |
|          | Micro F1  | 0.9273 |
| Neutral  | Precision | 0.9538  |
|          | Recall    | 0.9610|
|          | F1        | 0.9574 |
|          | Support   | 2513     |
| Negative | Precision | 0.7547 |
|          | Recall    | 0.6557 |
|          | F1        | 0.7018 |
|          | Support   | 61       |
| Positive | Precision | 0.67713 |
|          | Recall    | 0.61134 |
|          | F1        | 0.64255 |
|          | Support   | 494      |

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