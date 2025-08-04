#!/usr/bin/env bash

pip install --upgrade transformers optimum[exporters]

OUTPUT_DIR="$(pwd)/models"
mkdir -p "$OUTPUT_DIR"

# list of Hub model IDs to convert
MODELS=(
  "Alibaba-NLP/gte-modernbert-base"
  "Alibaba-NLP/gte-reranker-modernbert-base"

)
# ensure you have the right tools installed

for MODEL_ID in "${MODELS[@]}"; do
  NAME=$(basename "$MODEL_ID")
  DEST="$OUTPUT_DIR/$NAME"
  echo "➤ Exporting $MODEL_ID → $DEST …"

  optimum-cli export onnx \
    --model "$MODEL_ID" \
    --output "$DEST" \
    --opset 16 \
    --device cpu \
    --trust-remote-code \
    --task sentence-similarity
done

echo "[INFO] All models exported to $OUTPUT_DIR."

