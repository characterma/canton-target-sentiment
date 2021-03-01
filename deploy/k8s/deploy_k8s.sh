#!/bin/sh

# helm install or upgrade

if [ "$1" == "upgrade" ]; then
	
	# dry-run once before actual deployment
	./deploy_k8s_dryrun.sh upgrade
	
	echo "helm upgrade -i --debug --namespace $K8S_DEPLOY_NAMESPACE $RELEASE_NAME ./$CHART_NAME"
	helm upgrade -i --namespace $K8S_DEPLOY_NAMESPACE $RELEASE_NAME ./$CHART_NAME
else
	
	# dry-run once before actual deployment
	./deploy_k8s_dryrun.sh
	
	echo "helm install --namespace $K8S_DEPLOY_NAMESPACE -n $RELEASE_NAME ./$CHART_NAME"
	helm install $RELEASE_NAME ./$CHART_NAME --namespace $K8S_DEPLOY_NAMESPACE
fi
