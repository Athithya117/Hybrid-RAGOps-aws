docker compose down --volumes --remove-orphans
docker network rm raglaser_rag-net


helm upgrade --install my-app ./chart --atomic --cleanup-on-fail
