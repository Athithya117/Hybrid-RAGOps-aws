{{- define "onnx.fullname" -}}{{- printf "%s" .Release.Name -}}{{- end -}}
{{- define "onnx.labels" -}}app.kubernetes.io/name: onnx-embedder
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: Helm
{{- end -}}
{{- define "onnx.namespace" -}}{{- .Values.namespace | default .Release.Namespace -}}{{- end -}}
