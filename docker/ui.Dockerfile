FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir evidently==0.7.23 gcsfs==2026.6.0

EXPOSE 8000

CMD ["sh", "-c", "evidently ui --host 0.0.0.0 --workspace gs://${GCP_BUCKET_NAME}/evidently-workspace --port 8000"]
