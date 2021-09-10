# **Target sentiment analysis for WBI**

## API Specification

**End-point: /predict**

**Type: POST**

**Description**


## Input & output formats (general)


**Input Data**

| Field                    | Descriptions                                                                     |
|--------------------------|----------------------------------------------------------------------------------|
| organization             | The organization name.                                                           |
| source                   | The source or author.                                                            |
| pub_code                 |                                                                                  |
| headline                 |                                                                                  |
| content                  |                                                                                  |
| extended_target_keywords | A list of target keywords. Each keyword should appear in headline or in content. |

**Sample Input**

```json
{
    "organization": "保安局", 
    "source": "香江望神州", 
    "pub_code": "im_youtube_hk",
    "headline": "鄧炳強批612基金「臨解散都要撈油水」 將作調查 不點名批評黎智英是「主腦」",
    "content": "#國安法#\n撲滅罪行委員會8月27日開會，保安局局長鄧炳強在會後見記者",
    "target_keywords": ["保安局"]
}

```

**Output Data**
| Field     | Descriptions                                                          |
|-----------|-----------------------------------------------------------------------|
| sentiment | The sentiment output by model. ("positive", "neutral", or "negative") |
| scores    | The probabilities of all sentiments.                                  |
| score     | The probability of the output sentiment.                              |
| need_pr   | Indicating whether proof-reading is needed. (true or false)           |


**Sample Output**

```json
{
    "sentiment": "negative",
    "scores": {
        "neutral": 0.23816733062267303,
        "negative": 0.759811282157898,
        "positive": 0.002021389314904809
    },
    "score": 0.759811282157898,
    "need_pr": true
}
```

## API Load Test

**Description**

* K8S enviroment: 
  * cpu: 4000m
  * memory: 2G
  * maximum replicas: 1

**Results -- CPU only**

| Number of concurrent callers | Number of requests processed | Requests per second | Failure |
|----------------------------|------------------------------|---------------------|---------|-------------------------|----------------------------|----------------------------|
| 1                         |   53                       |          0.87      | 0       | 
| 3                         |   88                       |          1.41      | 0       | 
| 5                         |   101                       |          1.56      | 0       | 
| 10                         |   102                       |          1.58      | 0       | 

