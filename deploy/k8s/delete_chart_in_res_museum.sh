#!/bin/sh

echo "curl -X "DELETE" $HELM_CHART_REPO/api/charts/$CHART_NAME/$CHART_VERSION"
curl -X "DELETE" "$HELM_CHART_REPO/api/charts/$CHART_NAME/$CHART_VERSION" || true

exit 0
