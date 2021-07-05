# the result report can be found inside collections/newman

if ! [[ -z "${POSTMAN_VAR_FILE}" ]]; then
        POSTMAN_VAR="-d vars/$POSTMAN_VAR_FILE"
fi

echo "docker run --rm -v $PWD/collections:/etc/newman -t $DK_PUB/postman/newman_alpine33:3.9.4 run $POSTMAN_COLLECTION_FILE $POSTMAN_VAR --reporters cli,html"
docker run --rm -v $PWD/collections:/etc/newman -t $DK_PUB/postman/newman_alpine33:3.9.4 run $POSTMAN_COLLECTION_FILE $POSTMAN_VAR --reporters cli,html