
if ! [[ -z "${POSTMAN_VAR_FILE}" ]]; then
        POSTMAN_VAR="-d collections/vars/$POSTMAN_VAR_FILE"
fi

echo "newman run collections/$POSTMAN_COLLECTION_FILE $POSTMAN_VAR --reporters cli,html --reporter-html-export ./result/report.html --bail"
newman run collections/$POSTMAN_COLLECTION_FILE $POSTMAN_VAR --reporters cli,html --reporter-html-export ./result/report.html --bail