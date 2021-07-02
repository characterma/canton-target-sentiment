
#### Below no need to change unless it's necessary

# Verify variable values
env | grep -E 'RELEASE_NAME'

# helm delete, allow extra params, i.e. --purge
helm delete $RELEASE_NAME -n $K8S_DEPLOY_NAMESPACE

exit 0



