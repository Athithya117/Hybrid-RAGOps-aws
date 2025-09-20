# Decreasing max_length (tokens per request) impacts latency much more per request than decreasing batch size.

time bash -c 'for i in $(seq 1 20); do \
  curl -s -X POST "http://127.0.0.1:8005/embed" \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"$(yes "word" | head -n 2000 | tr "\n" " ")\",\"max_length\":50}" > /dev/null; \
done'


time bash -c 'for i in $(seq 1 20); do \
  curl -s -X POST "http://127.0.0.1:8005/embed" \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"$(yes "word" | head -n 2000 | tr "\n" " ")\",\"max_length\":500}" > /dev/null; \
done'


time bash -c 'for i in $(seq 1 200); do \
  curl -s -X POST "http://127.0.0.1:8005/embed" \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"$(yes "word" | head -n 2000 | tr "\n" " ")\",\"max_length\":50}" > /dev/null; \
done'
