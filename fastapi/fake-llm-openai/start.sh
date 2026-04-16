#!/bin/bash
set -e

source /usr/local/etc/profile.d/conda.sh
conda activate base

PORT=8080
if [ ! -z "$OPENBAYES_SERVING_PRODUCTION" ]; then
    PORT=80
fi

echo "Starting Fake LLM Server on port $PORT..."
exec uvicorn fake_llm_server:app --host 0.0.0.0 --port "$PORT"
