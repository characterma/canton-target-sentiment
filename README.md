# **Target-dependent sentiment analysis for Rolex**

## API Specification

**URL: /**

**Type: POST**

**Description**

1. Process target-guided Chinese text sentiment analysis using TGSAN model
2. Trinary model are supported. 

**Input Data**

- `language` ("chinese" or "english"): indicates the language of input text.
- `doclist`: contains one or multiple labelunits, same format as the output from KG, with four required fields for each labelunit.
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
          "unit_text": "蕭邦手錶一直是上層社會的寵愛之物，但長期性的應用過程中在所難免出現一些常見故障，假如腕錶遭受強烈的撞擊，會給腕錶導致表針掉下來的狀況。",
          "subject_index": [[0, 2], [58, 60]],
          "aspect_index": []
        }, 
        {
          "unit_index": [0,100],
          "unit_text": "同樣道理，在黑暗中以紫外 燈照射roger dubuis excalibur blacklight所發出的七彩光芒，在剔透的鏤通機芯映襯下，也顯得更具深度及迷人。",
          "subject_index": [[16, 21], [22, 28], [29, 38], [39, 49], [64, 66]],
          "aspect_index": []
        }, 
        {
          "unit_index": [0,13],
          "unit_text": "#cartier #美洲豹",
          "subject_index": [[1, 8]],
          "aspect_index": []
        }, 
        {
            "unit_index": [0,220],
            "unit_text": "🤩🤩🤩🤩🤩🤩🤩🤩蕭邦手錶一直是上層社會的寵愛之物，但長期性的應用過程中在所難免出現一些常見故障，假如腕錶遭受強烈的撞擊，會給腕錶導致表針掉下來的狀況。",
            "subject_index": [[16, 18], [74, 76]],
            "aspect_index": [], 
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
                    "unit_index": [0,67],
                    "unit_text": "蕭邦手錶一直是上層社會的寵愛之物，但長期性的應用過程中在所難免出現一些常見故障，假如腕錶遭受強烈的撞擊，會給腕錶導致表針掉下來的狀況。",
                    "subject_index": [[0,2],[58,60]],
                    "aspect_index": [],
                    "sentiment": "negative"
                },
                {
                    "unit_index": [0,100],
                    "unit_text": "同樣道理，在黑暗中以紫外 燈照射roger dubuis excalibur blacklight所發出的七彩光芒，在剔透的鏤通機芯映襯下，也顯得更具深度及迷人。",
                    "subject_index": [[16,21],[22,28],[29,38],[39,49],[64,66]],
                    "aspect_index": [],
                    "sentiment": "positive"
                },
                {
                    "unit_index": [0,13],
                    "unit_text": "#cartier #美洲豹",
                    "subject_index": [[1,8]],
                    "aspect_index": [],
                    "sentiment": "neutral"
                },
                {
                    "unit_index": [0,220],
                    "unit_text": "🤩🤩🤩🤩🤩🤩🤩🤩蕭邦手錶一直是上層社會的寵愛之物，但長期性的應用過程中在所難免出現一些常見故障，假如腕錶遭受強烈的撞擊，會給腕錶導致表針掉下來的狀況。",
                    "subject_index": [[16, 18], [74, 76]],
                    "aspect_index": [], 
                    "sentiment": "negative"
                }
            ]
        }
    ]
}
```