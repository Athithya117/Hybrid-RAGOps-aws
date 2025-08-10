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

{{- define "rag8s.chart" -}}
{{ printf "%s-%s" .Chart.Name .Chart.Version }}
{{- end -}}

{{- define "rag8s.labels" -}}
app.kubernetes.io/name: {{ include "rag8s.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion }}
helm.sh/chart: {{ include "rag8s.chart" . }}
{{- with .Values.global }}
{{- with .labels }}
{{ toYaml . | trim | indent 0 }}
{{- end }}
{{- end }}
{{- end -}}

{{- define "rag8s.selectorLabels" -}}
app.kubernetes.io/name: {{ include "rag8s.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "rag8s.podLabels" -}}
app.kubernetes.io/name: {{ include "rag8s.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: {{ default "app" .Values.component | quote }}
{{- end -}}

{{- define "rag8s.namespace" -}}
{{- if .Values.global.namespace }}
{{- .Values.global.namespace -}}
{{- else -}}
{{- .Release.Namespace -}}
{{- end -}}
{{- end -}}

{{- define "rag8s.irsaAnnotations" -}}
{{- $acc := .Values.aws.accountId | default "" -}}
{{- $role := .Values.iam.roleName | default "" -}}
{{- if and $acc $role }}
eks.amazonaws.com/role-arn: "arn:aws:iam::{{ $acc }}:role/{{ $role }}"
{{- end }}
{{- end -}}

{{- define "rag8s.mergeAnnotations" -}}
{{- $ctx := index . "ctx" -}}
{{- $key := index . "key" -}}
{{- $m :=  (index $ctx.Values $key) | default dict -}}
{{- if $m }}
{{ toYaml $m | trim }}
{{- end -}}
{{- end -}}

{{- define "rag8s.subresource" -}}
{{- $ctx := index . "ctx" -}}
{{- $name := index . "name" -}}
{{- printf "%s-%s" (include "rag8s.fullname" $ctx) $name }}
{{- end -}}
