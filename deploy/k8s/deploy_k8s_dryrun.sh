#!/bin/sh

# Auto-rename the chart folder according to CHART_NAME defined in env.sh
if [ -d "$CHART_NAME" ]; then
	echo "Folder $CHART_NAME found."
elif [ -d "chart" ]; then
	echo "Rename folder /chart to /$CHART_NAME as defined."
	mv chart $CHART_NAME
else
	echo "No expected chart folder found, pls check."
	exit 1
fi

# Define paths
AILAB_PATH=../docker/ailab
CONFIG_PATH=$CHART_NAME/configs/
mkdir -p $CONFIG_PATH
#rm -rf $CONFIG_PATH/*

sh -c "$COPY_TO_CHART_CONFIG" #define in env.sh

# Verify variable values
env | grep -E 'K8S_DEPLOY_NAMESPACE|APP_NAME|IMAGE_REPOS|IMAGE_TAG|RELEASE_NAME|CHART_NAME|CHART_VERSION|CI_REF'

# Replace chart name, version and description in Chart.yaml 
sed -i 's/^name:.*$/name: '"$CHART_NAME"'/' $CHART_NAME/Chart.yaml
sed -i 's/^version:.*$/version: '"$CHART_VERSION"'/' $CHART_NAME/Chart.yaml
sed -i 's/^description:.*$/description: '"$CHART_DESCRIPTION"'/' $CHART_NAME/Chart.yaml

# update chart dependencies
helm dep up ./$CHART_NAME

if [ "$1" == "upgrade" ]; then
   echo "helm upgrade --set ciBuildRef=$CI_REF --set appName=$APP_NAME --set imageRepository=$IMAGE_REPOS --set imageTag=$IMAGE_TAG 
   -i --debug --dry-run $RELEASE_NAME -n $K8S_DEPLOY_NAMESPACE ./$CHART_NAME"
   helm upgrade --set ciBuildRef=$CI_REF --set appName=$APP_NAME --set imageRepository=$IMAGE_REPOS --set imageTag=$IMAGE_TAG \
   -i --debug --dry-run $RELEASE_NAME -n $K8S_DEPLOY_NAMESPACE ./$CHART_NAME
else
   echo "helm install --set ciBuildRef=$CI_REF --set appName=$APP_NAME --set imageRepository=$IMAGE_REPOS --set imageTag=$IMAGE_TAG 
   --set    $RELEASE_NAME -n $K8S_DEPLOY_NAMESPACE --dry-run --debug ./$CHART_NAME"
   helm install --set ciBuildRef=$CI_REF --set appName=$APP_NAME --set imageRepository=$IMAGE_REPOS --set imageTag=$IMAGE_TAG \
      $RELEASE_NAME -n $K8S_DEPLOY_NAMESPACE --dry-run --debug ./$CHART_NAME
fi
