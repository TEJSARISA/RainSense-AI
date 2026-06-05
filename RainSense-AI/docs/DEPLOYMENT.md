# Deployment Guide

## Docker

```bash
docker build -t rainsense-ai .
docker run -p 8000:8000 rainsense-ai
```

For API and dashboard together:

```bash
cd docker
docker compose up --build
```

## Render

1. Create a new Web Service from the GitHub repository.
2. Use Python 3.11.
3. Set build command: `pip install -r requirements.txt && python train.py`.
4. Set start command: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`.

## Railway

1. Create a Railway project from the repository.
2. Add `PYTHONPATH=/app`.
3. Use start command: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`.

## AWS

Recommended path:

1. Build and push the Docker image to Amazon ECR.
2. Run the API on ECS Fargate or App Runner.
3. Store model artifacts in S3 for production retraining workflows.
4. Send prediction logs to CloudWatch.
5. Schedule `python retrain.py` with EventBridge.

