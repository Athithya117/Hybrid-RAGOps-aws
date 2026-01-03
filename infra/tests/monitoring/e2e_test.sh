# make lc
make deploy-vm 
make deploy-qdrant
make deploy-retriever
bash infra/generators/monitoring.sh --apply
bash infra/tests/monitoring/test_retriever.sh
bash infra/tests/monitoring/test_qdrant.sh

