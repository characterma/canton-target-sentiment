#!/bin/sh

SRC_ROOT="../.."
FOLDER_NAME="ailab"
echo $SRC_ROOT

# !!Require Changes!!: 
# =====================
# Copy necessary deployment files to /ailab

mkdir $FOLDER_NAME/
rsync -av $SRC_ROOT/src $FOLDER_NAME/src
rsync -av $SRC_ROOT/config $FOLDER_NAME/config
rsync -av $SRC_ROOT/models $FOLDER_NAME/models
rsync -av $SRC_ROOT/setup.py $FOLDER_NAME/setup.py
rsync -av $SRC_ROOT/start.sh $FOLDER_NAME/start.sh
rsync -av $SRC_ROOT/requirements.txt $FOLDER_NAME/requirements.txt


# =====================


#### Below no need to change unless it's necessary 


# docker login -u developer -p developer $DK_ESS
# docker login -u developer -p developer $DK_PUB
# docker rmi $IMAGE_REPOS:$IMAGE_TAG
# docker build -t $IMAGE_REPOS:$IMAGE_TAG .
# if [[ $? != 0 ]]; then
#     echo "build image error, please check."
#     exit 1
# fi
# docker push $IMAGE_REPOS:$IMAGE_TAG
# docker logout $DK_ESS
# docker logout $DK_PUB
