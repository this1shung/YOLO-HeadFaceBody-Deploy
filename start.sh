#!/bin/bash

tritonserver --model-repository=/models --http-port=8001 --grpc-port=8002 --log-verbose=1 &

# Wait for Triton to be ready (check HTTP health endpoint)
echo "Waiting for Triton Server to be ready..."
until curl -s http://localhost:8001/v2/health/ready > /dev/null; do
  echo "Triton not ready yet, waiting 2 seconds..."
  sleep 2
done
echo "Triton Server is ready!"

# Start FastAPI with uvicorn
exec uvicorn main:app --host 0.0.0.0 --port 8000