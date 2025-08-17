{{- /*
Common helper template functions
*/ -}}
{{- define "rag8s.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "rag8s.fullname" -}}
{{ printf "%s-%s" (include "rag8s.name" .) .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "rag8s.labels" -}}
app.kubernetes.io/name: {{ include "rag8s.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}
