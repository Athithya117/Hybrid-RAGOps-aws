# Decreasing max_length (tokens per request) impacts latency much more per request than decreasing batch size.

time bash -c 'for i in $(seq 1 20); do \
  curl -s -X POST "http://127.0.0.1:8005/embed" \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"The quick brown fox jumps over the lazy dog near the riverbank, while birds chirp cheerfully in the morning sunlight. Children play with colorful kites, and the gentle breeze carries the sweet scent of blooming flowers, creating a peaceful and joyful atmosphere for everyone present, enjoying nature together happily." | head -n 2000 | tr "\n" " ")\",\"max_length\":50}" > /dev/null; \
done'



yes "The quick brown fox jumps over the lazy dog near the riverbank, while birds chirp cheerfully in the morning sunlight. Children play with colorful kites, and the gentle breeze carries the sweet scent of blooming flowers, creating a peaceful and joyful atmosphere for everyone present, enjoying nature together happily" \
  | tr "\n" " " > /tmp/input.txt
echo "{\"text\":\"$(cat /tmp/input.txt)\",\"max_length\":50}" > /tmp/request.json
time bash -c 'for i in $(seq 1 20); do
  curl -s -X POST "http://127.0.0.1:8005/embed" \
       -H "Content-Type: application/json" \
       --data-binary @/tmp/request.json > /dev/null
done'




time bash -c 'for i in $(seq 1 20); do \
  curl -s -X POST "http://127.0.0.1:8005/embed" \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"$(yes "word" | head -n 2000 | tr "\n" " ")\",\"max_length\":500}" > /dev/null; \
done'


time bash -c 'for i in $(seq 1 200); do \
  curl -s -X POST "http://127.0.0.1:8005/embed" \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"$(yes "Building resilient retrieval systems demands modular design, clean data pipelines, vector embeddings optimized for your domain, and graph indexing for context linkage. Automate ingestion, transformation, and re-indexing tasks with dynamic Python orchestration, ensuring observability, scalability, and cost-aware performance tuning from day one." | tr "\n" " ")\",\"max_length\":10}" > /dev/null; \
done'

time bash -c 'for i in $(seq 1 20); do \
  curl -s -X POST "http://127.0.0.1:8005/embed" \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"$(yes "word" | head -n 2000 | tr "\n" " ")\",\"max_length\":50}" > /dev/null; \
done'

time bash -c 'for i in $(seq 1 20); do \
  curl -s -X POST "http://127.0.0.1:8005/embed" \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"$(yes "word" | head -n 2000 | tr "\n" " ")\",\"max_length\":25}" > /dev/null; \
done'

time bash -c 'for i in $(seq 1 200); do \
  curl -s -X POST "http://127.0.0.1:8005/embed" \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"$(yes "word" | head -n 2000 | tr "\n" " ")\",\"max_length\":25}" > /dev/null; \
done'


# test containers locally
# curl -s -X POST "http://127.0.0.1:8008/embed" -H "Content-Type: application/json" -d '{"text":"hello world","max_length":80}'
 

# python3 indexing_pipeline/ray_indexing_pipeline.py --libreoffice-ready-timeout 30

 # ---- AMD64  ----
docker run --rm -d --name embedder-test-run21-amd64 \
  -p 8005:8005 \
  --shm-size=1.8g \
  -v /workspace/models:/workspace/models:ro \
  -e PORT=8005 \
  athithya324/embedder-cpu-inference@sha256:c3c290f22318586071efa3d615d4e142cbf173a66b04f183fcd4e19808914026 && \
sleep 10 && \
curl -s -X POST "http://127.0.0.1:8005/embed" -H "Content-Type: application/json" -d '{"text":"hello world","max_length":80}'



# ---- ARM64 ----
docker run --rm -d --name embedder-test-run21-arm64 \
  -p 8006:8006 \
  --shm-size=1.8g \
  -v /workspace/models:/workspace/models:ro \
  -e PORT=8006 \
  athithya324/embedder-cpu-inference@sha256:61441535a6436f92a5d41832b5683716b87bf9323ea5defb1142db8bb5297340

# ---- GPU image but force CPU mode ----

ray stop && \
ray start --head --port=6380 && \
docker run -d --name embedder-test-run21-gpu-cpu \
  -p 8008:8008 \
  --shm-size=5g \
  -v /workspace/models:/workspace/models:ro \
  -e PORT=8008 \
  -e FORCE_CPU=1 \
  -e REPLICA_GPUS=0 \
  athithya324/indexing-embedder-gpu@sha256:4b238ceff0e42dda21373585c4ea174b4e4541f04ace61363ef1be5b52a7ddac && \
sleep 10 && \
curl -s -X POST "http://0.0.0.0:8008/embed" -H "Content-Type: application/json" -d '{"text":"hello world","max_length":80}'
