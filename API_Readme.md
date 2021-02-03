# Request 
## Fields

* "headline": headline of post (text)
* "content": content of post (text)
* "start_ind": the starting position of target (integer)
* "end_ind": the ending position of target (integer) 
* "target_in_hl": whether the target is in headline or content (integer, 0 for content, 1 for headline)
* "all_in_content_fmt": a special format for parsing (integer, please use 0 or skip)

## Example
```json
{
    "headline": "大家最近買左D乜 分享下?", 
    "content": "ABC, sales 態度麻麻", 
    "start_ind": 0, 
    "end_ind": 3, 
    "target_in_hl": 0, 
    "all_in_content_fmt": 0
}
```

# Respose
## Fields

* "data": sentiment (text, "neutral", "positive", or "negative")
* "message": OK means success (text)

## Example
{
    "data": "negative",
    "message": "OK"
}