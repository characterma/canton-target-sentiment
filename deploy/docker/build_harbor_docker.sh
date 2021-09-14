#!/bin/sh

SRC_ROOT=../..

# !!Require Changes!!: 
# =====================
# Copy necessary deployment files to /ailab

rsync -av $SRC_ROOT/src/ ailab/src/
rsync -av $SRC_ROOT/output/ ailab/output/

# =====================

#### Below no need to change unless it's necessary 
docker login -u 'ci-headless-user' -p 'ci1234' $HARBOR_REPO

docker rmi $HARBOR_IMAGE_REPOS:$IMAGE_TAG
docker build -t $HARBOR_IMAGE_REPOS:$IMAGE_TAG .
if [[ $? != 0 ]]; then
    echo "build image error, please check."
    exit 1
fi
docker push $HARBOR_IMAGE_REPOS:$IMAGE_TAG
docker logout $HARBOR_IMAGE_REPOS
