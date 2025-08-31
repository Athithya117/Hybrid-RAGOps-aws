FROM ubuntu:22.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3.10 python3-pip build-essential ca-certificates libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY embedding_model_requirements-cpu.txt /app/embedding_model_requirements-cpu.txt
RUN python3.10 -m pip install --upgrade pip setuptools wheel --no-cache-dir && python3.10 -m pip install -r /app/embedding_model_requirements-cpu.txt --no-cache-dir
COPY host_embedding_model_local.py /app/host_embedding_model_local.py

FROM ubuntu:22.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3.10 python3-pip libgomp1 ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /app /app
ENV HOST=0.0.0.0 PORT=8000 MODEL_DIR=/workspace/models/onnx/gte-modernbert-base-onnx-int8 ROUTE_PREFIX=/
EXPOSE 8000
CMD ["python3.10","/app/host_embedding_model_local.py"]


# docker build -t embedder:dev .
# docker run --rm -p 8000:8000 \
#   -v /workspace/models:/workspace/models:ro \
#  -v $(pwd):/app \
#  -e HOST=0.0.0.0 -e PORT=8000 \
#  embedder:dev

