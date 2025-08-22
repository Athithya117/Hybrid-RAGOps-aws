FROM python:3.10-slim
WORKDIR /app
COPY main.py .
RUN pip install fastapi uvicorn
EXPOSE 3000
CMD ["python","main.py"]
