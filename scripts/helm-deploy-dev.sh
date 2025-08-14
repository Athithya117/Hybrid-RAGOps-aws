export CHART_DIR=./infra/charts/rag8s-aws        
export VALUES=$CHART_DIR/values.yaml
export RELEASE=my-rag8s
export NAMESPACE=rag8s-system


helm lint $CHART_DIR --values $VALUES


cd infra/charts/rag8s-aws
helm dependency update

