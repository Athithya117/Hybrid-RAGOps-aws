
export PYTHONPATH=$(pwd)
export S3_BUCKET=e2e-rag-system16
export IS_MULTILINGUAL=true          # or false if your data is english only  
export S3_RAW_PREFIX=data/raw/    
export S3_CHUNKED_PREFIX=data/chunked/
export CHUNK_FORMAT=json              # or jsonl for storage efficieny during headless mode       
export RAPIDOCR_CACHE="/home/user/RAG8s/models/" 
export DOCLING_ARTIFACTS_PATH="/home/user/RAG8s/models/"
export PDF_PARSER="docling"           # or extractous if a faster parser is required and if pdfs doesnt have complex layout/tables
export PDF_TABLE_MODE="accurate"     # or fast (ignore if not using Docling)
export LOG_LEVEL=INFO