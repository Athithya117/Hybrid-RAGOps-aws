{{- /*
Helpers for rag8s chart
*/ -}}

{{- define "rag8s.name" -}}
{{- default .Chart.Name .Values.nameOverride -}}
{{- end -}}

{{- define "rag8s.fullname" -}}
{{- $default := printf "%s-%s" .Release.Name (include "rag8s.name" .) -}}
{{- $name := default $default .Values.fullnameOverride -}}
{{- if gt (len $name) 63 -}}
{{- $name = (printf "%.63s" $name) -}}
{{- end -}}
{{- trimSuffix "-" $name -}}
{{- end -}}

{{- define "rag8s.labels" -}}
app.kubernetes.io/name: {{ include "rag8s.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}
