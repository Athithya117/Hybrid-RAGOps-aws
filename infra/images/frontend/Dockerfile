FROM python:3.13-slim-bookworm AS build
WORKDIR /app
RUN apt-get update && apt-get install --no-install-recommends -y build-essential curl && rm -rf /var/lib/apt/lists/*
COPY requirements-cpu.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-cpu.txt

FROM python:3.13-slim-bookworm AS run
WORKDIR /app
COPY --from=build /install /usr/local
COPY . .
CMD ["streamlit", "run", "streamlit_ui.py", "--server.port=3000", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
