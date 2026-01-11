# apps/index/Dockerfile
# Final application image. This expects a prebuilt base image to be supplied via --build-arg.
# If you want to build the base locally, use your Dockerfile.base and build/push that separately.
ARG BASE_IMAGE=indexing-base:2026-01-10
FROM ${BASE_IMAGE} AS app

LABEL org.opencontainers.image.title="indexing-pipeline"
LABEL org.opencontainers.image.base="${BASE_IMAGE}"

# Copy application code only (keeps rebuilds fast)
# Also copy the ulimit helper so build script can run it on the host or operator can run it manually.
COPY . /indexing_pipeline
COPY ulimit_increase.sh /usr/local/bin/ulimit_increase.sh

WORKDIR /indexing_pipeline

# Ensure pre_conversions script is executable if present
RUN if [ -f /indexing_pipeline/pre_conversions.py ]; then chmod +x /indexing_pipeline/pre_conversions.py || true; fi

# Make ulimit helper executable (for operators / build script)
RUN if [ -f /usr/local/bin/ulimit_increase.sh ]; then chmod +x /usr/local/bin/ulimit_increase.sh || true; fi

# Conservative runtime defaults (inherited from base but re-exposed here for clarity)
ENV OVERWRITE_ALL_AUDIO_FILES=false
ENV OVERWRITE_OTHER_TO_PDF=true
ENV OVERWRITE_SPREADSHEETS_WITH_CSV=false
ENV STORAGE_PROTOCOL=az
ENV AZURE_CONTAINER=rag-data-prod
ENV AZURE_CHUNKED_PREFIX=data/chunked/
ENV AZURE_ENDPOINT_SUFFIX=core.windows.net
ENV USE_MANAGED_IDENTITY=0
ENV CHUNK_FORMAT=json
ENV MAX_TOKENS_PER_CHUNK=400
ENV MIN_TOKENS_PER_CHUNK=100
ENV NUMBER_OF_OVERLAPPING_SENTENCES=2
ENV TOKEN_ENCODER=cl100k_base

# Ensure PATH / PYTHONPATH remain correct
ENV PATH=/opt/venv/bin:$PATH
ENV PYTHONPATH=/indexing_pipeline

# Note: We intentionally do NOT run ulimit/sysctl inside the image build.
# The accompanying build script will attempt to apply host ulimit changes prior to building if configured.
