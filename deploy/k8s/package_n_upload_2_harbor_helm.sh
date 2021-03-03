#!/bin/sh

# dry-run once
./deploy_k8s_dryrun.sh upgrade

# Verify variable values
env | grep -E 'CHART_VERSION|HELM_CHART_REPO'

PACKAGE_DIR=./pkg
mkdir -p $PACKAGE_DIR
rm -rf $PACKAGE_DIR/*

echo "helm package --destination $PACKAGE_DIR ./$CHART_NAME"
helm package --destination $PACKAGE_DIR ./$CHART_NAME

echo "helm push "@$PACKAGE_DIR/$CHART_NAME-$CHART_VERSION.tgz" ai-repo --username ci-headless-user --password ci1234"

helm push "$PACKAGE_DIR/$CHART_NAME-$CHART_VERSION.tgz" ai-repo --username "ci-headless-user" --password "ci1234"
