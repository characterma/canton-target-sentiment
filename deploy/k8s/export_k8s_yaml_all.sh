#!/bin/sh

# dry-run once before actual deployment
sh ./deploy_k8s_dryrun.sh upgrade

# helm template export yaml files, use default namespace (otherwise set --namespace $K8S_DEPLOY_NAMESPACE , similarily set -n RELEASE_NAME)
echo "helm template ./$CHART_NAME > deploy-k8s.yaml"
helm template ./$CHART_NAME > deploy-k8s.yaml

