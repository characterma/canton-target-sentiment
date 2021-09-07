#!/bin/sh

# helm install or upgrade

if [ "$1" == "upgrade" ]; then
	
	# dry-run once before actual deployment
	./deploy_k8s_dryrun.sh upgrade
	
	echo "helm upgrade --set ciBuildRef=$CI_REF --set appName=$APP_NAME --set imageRepository=$IMAGE_REPOS --set imageTag=$IMAGE_TAG -i --debug --namespace $K8S_DEPLOY_NAMESPACE $RELEASE_NAME ./$CHART_NAME"
	helm upgrade --set ciBuildRef=$CI_REF --set appName=$APP_NAME --set imageRepository=$IMAGE_REPOS --set imageTag=$IMAGE_TAG \
    -i --namespace $K8S_DEPLOY_NAMESPACE $RELEASE_NAME ./$CHART_NAME
else
	
	# dry-run once before actual deployment
	./deploy_k8s_dryrun.sh
	
	echo "helm install --set ciBuildRef=$CI_REF --set appName=$APP_NAME --set imageRepository=$IMAGE_REPOS --set imageTag=$IMAGE_TAG --namespace $K8S_DEPLOY_NAMESPACE $RELEASE_NAME ./$CHART_NAME"
	helm install --set ciBuildRef=$CI_REF --set appName=$APP_NAME --set imageRepository=$IMAGE_REPOS --set imageTag=$IMAGE_TAG \
    --namespace $K8S_DEPLOY_NAMESPACE $RELEASE_NAME ./$CHART_NAME
fi
