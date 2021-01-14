#!/bin/sh

SRC_ROOT=../..
echo $(ls ailab/)

# !!Require Changes!!: 
# =====================
# Copy necessary deployment files to /ailab

rsync -av $SRC_ROOT/src/ ailab/src/
echo $(ls ailab/)
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
