# **Target-dependent sentiment analysis for Rolex**

## API Specification

**URL: /rolex_sentiment**

**Type: POST**

**Description**

1. Process target-guided sentiment analysis.
2. Chinese model are currently supported. 

**Input Data**

- `language` ("chinese" or "english"): indicates the language of input text.
- `doclist`: contains one or multiple documents
- `labelunits`: contains labelunits in each document. Each labelunit has four required fields.
  * unit_index
  * unit_text
  * subject_index
  * aspect_index

**Sample Input**

```json
{
  "language": "chinese",
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
  * sentiment ("neutral", "negative", or "positive")

**Sample Output**

```json
{
    "language": "chinese",
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