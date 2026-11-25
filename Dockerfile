FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY api.py .
# Vector DB lives here; mount a persistent volume at /data in production
ENV CHROMA_PATH=/data/chroma_db
RUN mkdir -p /data
EXPOSE 8000
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
