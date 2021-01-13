{{/* vim: set filetype=mustache: */}}
{{/*
Define common metadata
*/}}
{{- define "commonMeta" }}
  name: {{ .Values.appName }}
  labels:
    app: {{ .Values.appName }}
    chart: {{ .Chart.Name }}-{{ .Chart.Version | replace "+" "_" }}
    release: {{ .Release.Name }}
    heritage: {{ .Release.Service }}
    ci-build-ref: {{ .Values.ciBuildRef | default "un-known" | substr 0 8 }}
{{- end }}
