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


# !!Require Changes!!:
# =====================
# Copy all necessary files (i.e. config/script) to /configs for creating K8S ConfigMap

#cp $AILAB_PATH/conf/gunicorn.conf $CONFIG_PATH
#cp $AILAB_PATH/conf/logging.conf $CONFIG_PATH



#cp $AILAB_PATH/conf/config.ini $CONFIG_PATH
# note: converted config.ini to .tpl inside chart/configs/tpl/

# =====================



# Verify variable values
env | grep -E 'K8S_DEPLOY_NAMESPACE|APP_NAME|IMAGE_REPOS|IMAGE_TAG|RELEASE_NAME|CHART_NAME|CHART_VERSION|CI_REF'

# Replace chart name, version and description in Chart.yaml
sed -i 's/^name:.*$/name: '"$CHART_NAME"'/' $CHART_NAME/Chart.yaml
sed -i 's/^version:.*$/version: '"$CHART_VERSION"'/' $CHART_NAME/Chart.yaml
sed -i 's/^description:.*$/description: '"$CHART_DESCRIPTION"'/' $CHART_NAME/Chart.yaml

# Record CI_BUILD_REF in order to back-reference to specific gitlab commit
sed -i 's/^ciBuildRef.*$/ciBuildRef: '"$CI_REF"'/' $CHART_NAME/values.yaml
sed -i 's/^  ciBuildRefProj.*$/  ciBuildRefProj: '"$CI_REF"'/' $CHART_NAME/values.yaml
sed -i 's/^appName.*$/appName: '"\"$APP_NAME\""'/' $CHART_NAME/values.yaml
sed -i 's/^imageRepository.*$/imageRepository: '"$(echo $IMAGE_REPOS | sed -e 's/[\/&]/\\&/g')"'/' $CHART_NAME/values.yaml
sed -i 's/^imageTag.*$/imageTag: '"\"$IMAGE_TAG\""'/' $CHART_NAME/values.yaml

# update chart dependencies
helm dep up ./$CHART_NAME

if [ "$1" == "upgrade" ]; then
   echo "helm upgrade -i --debug --dry-run $RELEASE_NAME -n $K8S_DEPLOY_NAMESPACE ./$CHART_NAME"
   helm upgrade -i --debug --dry-run $RELEASE_NAME -n $K8S_DEPLOY_NAMESPACE ./$CHART_NAME
else
   echo "helm install $RELEASE_NAME -n $K8S_DEPLOY_NAMESPACE --dry-run --debug ./$CHART_NAME"
   helm install $RELEASE_NAME -n $K8S_DEPLOY_NAMESPACE ./$CHART_NAME --dry-run --debug
fi
