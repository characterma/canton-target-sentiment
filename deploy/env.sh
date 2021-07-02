
# !!Require Changes!!:
# =====================
# "GROUP_NAME-APP_NAME" will be used as Chart Name

# refer to GIT Lab project group, use the code within the bracket under group description
# e.g.: for TopicClassification > API > topic-tagging-api the group name will be "topcls".  The GIT project group description is "(topcls) Topic classification projects"
GROUP_NAME=senti

# This will be used for image and K8S module / ingress name
# If GIT Lab project name is different from deployed module on K8S, please use K8S module name instead
APP_NAME=canton-target-sentiment

# Update Image Tag if different from any deployed version
IMAGE_TAG=1.1

# Update Chart Tag if different from any deployed version
CHART_VERSION=0.1.1

CHART_DESCRIPTION="Cantonese target sentiment for Rolex."

# Optional, for running deployment script manually
K8S_DEPLOY_NAMESPACE=playground



### Set below Variables ONLY for creating Standalone Container on docker Server via docker/run_docker.sh  ### 

## Usually define for running on one of DEV servers (ess37~39): 

# Set container deployment server
DOCKER_DEPLOY_SERVER="ess38"
# Set docker container name, naming standard: (project|<user_name>)-<app name>
DOCKER_DEPLOY_CONTAINER_NAME="$GITLAB_USER_LOGIN-$APP_NAME"
# Define docker port mappings. multiple mappings can be defined like: "-p 9090:8080 -p 80:80 -p 20000~20005:10000~10005"
DOCKER_DEPLOY_CONTAINER_PORT_PARAMS="-p 8091:8080"
# Optional, set environment variables inside container, i.e. "-e TZ=Asia/Hong_Kong"
DOCKER_ENV_VARS=
# Optional, set volume mount from host to container, i.e. "-v /tmp/dummyfile:/ailab/dummyfile"
DOCKER_VOLUMES_MOUNT=
# Optional, set container start command
DOCKER_CMD=


### Set Postman Testing ###

# Postman collection file name (DEFAULT: postman_collection.json)
POSTMAN_COLLECTION_FILE="postman_collection.json"

# Optional, to override collection's variables from file inside test/collections/variables/ (i.e. playground.json)
POSTMAN_VAR_FILE=

# =====================

# Below are predefined. Don't Change unless you know what to achieve
CHART_NAME=$GROUP_NAME-$APP_NAME

IMAGE_REPOS=$DK_ESS/$GROUP_NAME/$APP_NAME

RELEASE_NAME=$K8S_DEPLOY_NAMESPACE-$CHART_NAME

HELM_CHART_REPO=http://ess-deploy.wisers.com:8585

CI_REF=${CI_BUILD_REF:-un-known}

CHART_DESCRIPTION="(${CI_REF:0:8}) $CHART_DESCRIPTION"

# check if it's in git tagging flow
if [[ -z "${CI_COMMIT_TAG}" ]]; then
  echo "This is GIT development flow"
  IMAGE_TAG="$IMAGE_TAG-dev"
  CHART_VERSION="$CHART_VERSION-dev"
else
  echo "This is GIT tagging flow, tagName=$CI_COMMIT_TAG"  
  if echo $CI_COMMIT_TAG | grep -qE "^stable-.*$" ; then
  	IMAGE_TAG="$IMAGE_TAG-stable"
  	CHART_VERSION="$CHART_VERSION-stable"
  elif echo $CI_COMMIT_TAG | grep -qE "^release-.*$" ; then
 	IMAGE_TAG="$IMAGE_TAG-release"
 	CHART_VERSION="$CHART_VERSION-release"
  else
  	echo "Non-standard tag name: $CI_COMMIT_TAG, please revise!"
  	# exit error to terminiate the ci flow
  	exit 1	
  fi		 
fi


export K8S_DEPLOY_NAMESPACE APP_NAME CHART_NAME IMAGE_REPOS IMAGE_TAG RELEASE_NAME CHART_VERSION HELM_CHART_REPO \
CHART_DESCRIPTION DOCKER_DEPLOY_SERVER DOCKER_DEPLOY_CONTAINER_NAME DOCKER_DEPLOY_CONTAINER_PORT_PARAMS CI_REF \
DOCKER_ENV_VARS DOCKER_VOLUMES_MOUNT DOCKER_CMD POSTMAN_COLLECTION_FILE POSTMAN_VAR_FILE

if command -v dos2unix > /dev/null ; then
  find . -type f | xargs dos2unix -q
fi

# ensure some paths exist
echo $(ls)
echo $(find . -name env.sh)
DEPLOY_DIR=$(dirname $(find . -name env.sh))
mkdir -p $DEPLOY_DIR/docker/ailab
mkdir -p $DEPLOY_DIR/k8s/chart/configs/tpl/ 
HARBOR_REPO="harbor.wisers.com"

HARBOR_PROJECT_NAME="ailab"

HARBOR_IMAGE_REPOS=$HARBOR_REPO/$HARBOR_PROJECT_NAME/$APP_NAME

export HARBOR_REPO HARBOR_PROJECT_NAME HARBOR_IMAGE_REPOS

HARBOR_REPO="harbor.wisers.com"

HARBOR_PROJECT_NAME="ailab"

HARBOR_IMAGE_REPOS=$HARBOR_REPO/$HARBOR_PROJECT_NAME/$APP_NAME

export HARBOR_REPO HARBOR_PROJECT_NAME HARBOR_IMAGE_REPOS
