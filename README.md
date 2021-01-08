# Cantonese Target Dependent Sentiment Analysis

## Deployment Guide

### Setup API with Docker

1. Build docker image.
```bash
cd ./deploy/docker
sh build_docker.sh
docker build -t IMAGE_NAME . 
```
2. Start a container.
```bash
docker run --name CONTAINER_NAME -m 8GB -cpus 4 -p PORT:8080 -td IMAGE_NAME
```

3. Start API. (**!!!!!**) API_NAME needs to be 'sanic' or 'aiohttp'.
```
docker exec -d CONTAINER_NAME sh start.sh API_NAME
```

### Load Testing 
1. Setup API with docker
2. Run Locust tests (for specific tests).
```bash
docker exec -d CONTAINER_NAME python ./deploy/test/locustfile.py -u=CONCORRENT_USERS --api=API_NAME
```
3. Or, run pre-defined Locust tests.
```bash
docker exec -d CONTAINER_NAME bash -c "cd ./deploy/test/ ; bash run_all_test.sh API_NAME ; cd ../.."
```
4. Retrieve test results
```bash
docker cp CONTAINER_NAME:./deploy/test/ ./
```

### API Usage 

1. POST to `http://HOST:PORT/target_sentiment` with body as:
```json
{
    "content": "## Headline ##\n大家最近買左D乜 分享下?\n## Content ##\n引用:\n原帖由 綠茶寶寶 於 2018-12-31 08:40 PM 發表\n買左3盒胭脂\nFit me 胭脂睇youtuber推介話好用，用完覺得麻麻\n原來fit me麻麻 我買左YSL 支定妝噴霧 用完覺得無想像咁好\n\n", 
    "start_ind": 141, 
    "end_ind": 144
}
```
2. Response:
```json
{
    "data": "negative",
    "message": "OK"
}

```
