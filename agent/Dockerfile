# Dockerfile — cloud server deployment (Tencent Cloud Lighthouse / any Linux host)
#
# The container listens on 7860 internally; map it to any host port:
#   docker build -t isaac-agent .
#   docker run -d --name isaac-agent --restart=always \
#     -p 8765:7860 --env-file /root/isaac.env isaac-agent
#
# China-network friendly: apt + pip both use Tsinghua mirrors, so the build
# is fast on domestic servers. For pulling the base image itself, configure
# the Docker registry mirror on the HOST (see DEPLOY_TENCENT.md).

FROM python:3.11-slim

# Switch apt sources to Tsinghua mirror (fast in China).
# Debian bookworm+ uses deb822 format (debian.sources); older uses sources.list.
RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null; \
    sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list 2>/dev/null; \
    true

# Install CJK fonts (matplotlib needs them for Chinese labels in plots)
RUN apt-get update && apt-get install -y --no-install-recommends \
        fonts-noto-cjk \
        fonts-noto-color-emoji \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better build caching.
# Two batches: (1) skill-side runtime deps used by scripts/* (jinja2 templates,
# yaml configs, tensorboard event parsing, numpy stats, matplotlib plots);
# (2) agent_server's own deps (fastapi/httpx/uvicorn...).
COPY agent_server/requirements.txt /app/agent_server/requirements.txt
RUN pip install --no-cache-dir \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
        jinja2 pyyaml numpy matplotlib tensorboard \
 && pip install --no-cache-dir \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
        -r /app/agent_server/requirements.txt

# Copy agent_server code
COPY agent_server/ /app/agent_server/

# Copy skill assets that agent_server depends on (scripts, resources, templates,
# references, test_data, example env). agent_server imports scripts.* at runtime.
COPY scripts/ /app/scripts/
COPY resources/ /app/resources/
COPY templates/ /app/templates/
COPY references/ /app/references/
COPY tests/test_data/ /app/tests/test_data/
COPY examples/ /app/examples/

# Configure matplotlib to use Noto CJK by default (avoids per-call setup)
ENV MPLBACKEND=Agg \
    MATEMBEDDING_FONTPATH=/usr/share/fonts/opentype/noto

# HF Spaces runtime config (Secrets are injected at container start)
ENV HOST=0.0.0.0 \
    PORT=7860 \
    SKILL_ROOT=/app \
    PYTHONPATH=/app:/app/agent_server \
    FILES_DIR=/app/agent_server/logs/files

# Create files directory for runtime-generated attachments
RUN mkdir -p /app/agent_server/logs/files

WORKDIR /app/agent_server

EXPOSE 7860

# Healthcheck — agent should respond within 30s of startup
HEALTHCHECK --interval=60s --timeout=10s --start-period=40s --retries=2 \
    CMD curl -fsS http://localhost:7860/health || exit 1

# Run the agent server
CMD ["python", "main.py"]
