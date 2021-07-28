#!/bin/sh

# Change the .gitlab-ci.yml#
sed -i 's/--purge/ /g' .gitlab-ci.yml
echo "remove '--purge' in .gitlab-ci.yml"

# Change the deploy/k8s/chart/Chart.yaml#
sed -i 's/apiVersion: v1/apiVersion: v2/g' deploy/k8s/chart/Chart.yaml
echo "change chart version from v1 to v2"

# Change the deploy/k8s/deploy_k8s.sh#
sed -i 's/-n \$RELEASE_NAME/\$RELEASE_NAME/g' deploy/k8s/deploy_k8s.sh
echo "remove '-n' from deploy_k8s.sh"

# Change the deploy/k8s/package_n_upload_2_harbor_helm.sh#
sed -i 's/--save=false/ /g' deploy/k8s/package_n_upload_2_harbor_helm.sh
echo "remove '--save=false' in package_n_upload_2_harbor_helm.sh"
sed -i 's/--save=false/ /g'  deploy/k8s/package_n_upload_2_res_museum.sh
echo "remove '--save=false' in package_n_upload_2_res_museum.sh"


# Change the deploy/k8s/deploy_k8s_dryrun.sh#
sed -i 's/-n/ /g' deploy/k8s/deploy_k8s_dryrun.sh
echo "remove '-n' in deploy_k8s_dryrun.sh"
sed -i 's/\$RELEASE_NAME/\$RELEASE_NAME -n \$K8S_DEPLOY_NAMESPACE/g' deploy/k8s/deploy_k8s_dryrun.sh
echo "add the namespace in deploy_k8s_dryrun.sh"

# Change the deploy/k8s/undeploy_k8s.sh#
sed -i 's/\$RELEASE_NAME/\$RELEASE_NAME -n \$K8S_DEPLOY_NAMESPACE/g' deploy/k8s/undeploy_k8s.sh
echo "add the namespace in undeploy_k8s.sh"

# kill itself#
echo "remove changeforhelm3.sh"
rm -f changeforhelm3.sh
