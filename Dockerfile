# ==============================================================
# LLM Behavioral Monitor - Docker Image
# ==============================================================
# Build:
#   docker build -t llm-monitor .
#
# Run (GPU, models mounted from host):
#   docker run --gpus all -p 8000:8000 \
#     -v /path/to/models:/models \
#     llm-monitor --model Qwen_Qwen3-4B --device cuda:0
#
# Run (CPU / simulated mode, no model mount):
#   docker run -p 8000:8000 \
#     llm-monitor --device cpu
# ==============================================================

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git \
    && rm -rf /var/lib/apt/lists/*

# Set python3 as default
RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY core/ core/
COPY *.py ./

# Copy probe files
COPY trained_probes_qwen3_4b.pkl ./
COPY trained_probes_qwen3_4b.json ./
COPY trained_probes/ trained_probes/

# Default model directory (mount from host at runtime)
ENV DASHBOARD_MODEL_DIR=/models
ENV COURT_MODEL_DIR=/models

# Single port — launcher serves all sub-apps (court + dashboard)
EXPOSE 8000

# Health check against the launcher's status endpoint
HEALTHCHECK --interval=30s --timeout=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/status')" || exit 1

ENTRYPOINT ["python", "launcher.py", "--model-dir", "/models"]
CMD ["--model", "Qwen_Qwen2.5-0.5B-Instruct", "--device", "cuda:0"]
