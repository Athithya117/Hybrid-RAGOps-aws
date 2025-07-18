


export AWS_ACCESS_KEY_ID=
export AWS_SECRET_ACCESS_KEY=
export AWS_REGION=ap-south-1
export S3_BUCKET_NAME=e2e-rag-system16
export PYTHONPATH=$(pwd)


aws s3 ls $S3_BUCKET_NAME/data/raw/





{
  "id": "chunk_{sha256}_{chunk_index}",
  "embedding": [],
  "payload": {
    "document_id": "{sha256}",
    "chunk_id": "chunk_{chunk_index}",
    "chunk_index": 0,
    "text": "Text content or transcript here.",
    "parser": "paddleocr + layoutLM + python-docx",
    "pipeline_stage": "embedded",
    "source": {
      "path": "s3://bucket/data/raw/file.docx",
      "hash": "sha256:abc123...",
      "file_type": "docx",
      "page_number": null,
      "start_time": null,
      "end_time": null,
      "line_range": [0, 5],
      "section_title": null
    },
    "metadata": {
      "language": "en",
      "is_multilingual": false,
      "is_ocr": false,
      "chunk_type": "paragraph" | "page" | "audio_segment",
      "timestamp": "2025-07-01T00:00:00Z",
      "tags": []
    },
    "entities": [],
    "triplets": []
  }
}


