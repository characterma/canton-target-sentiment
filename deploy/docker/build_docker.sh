#!/bin/sh
SRC_ROOT=../..

# !!Require Changes!!: 
# =====================
# Copy necessary deployment files to /ailab

# Create (if not exist) ailab/ and Copy everything under ./src/ to ailab/
# E.g. ./src/run.py to ailab/run.py
rsync -av $SRC_ROOT/src/ ailab/src/
rsync -av $SRC_ROOT/model/ ailab/model/
# =====================


#### Below no need to change unless it's necessary 


docker login -u developer -p developer $DK_ESS
docker login -u developer -p developer $DK_PUB
docker rmi $IMAGE_REPOS:$IMAGE_TAG
docker build -t $IMAGE_REPOS:$IMAGE_TAG .
if [[ $? != 0 ]]; then
    echo "build image error, please check."
    exit 1
fi
docker push $IMAGE_REPOS:$IMAGE_TAG
docker logout $DK_ESS
docker logout $DK_PUB
