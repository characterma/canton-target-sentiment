# Cantonese SA

# Usage
## Installation
```
python setup.py develop
```
## Train Model
1. Edit configuations under `./config/`
1. Run the following command in terminal. (install any needed library missing)
```
python main.py --do_train
```
# Using Docker image

## Starting a container
1. ssh a GPU server 
1. cd to your directory (such as `/home/developer/Users/tonychan`)
1. Pull image `canton-sa-gpu` (not yet there)
1. Run the follwing command:
```
docker run --itd --name <container_name> -p 28080:8080 -p 28885:8885 -p 28886:8886 -p 28887:8887 -p 28888:8888 -p 28889:8889 -v -v "$PWD":/root --privileged canton-sa-gpu 
```

## Entering a container
```
docker attach <container_name>
```

## Exiting a container (without stopping it)
`Ctrl+P` + `Ctrl+Q`


# Reference
- Some functions and program designs are borrowed from https://github.com/monologg/R-BERT.
- Model is similar to TDBERT in "Target-Dependent Sentiment Classification with BERT".