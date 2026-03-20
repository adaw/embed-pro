FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY embed-pro.py .

# Non-root user
RUN useradd -m -u 1000 app
USER app

ENV PORT=8020
ENV PRELOAD=true
ENV PYTHONUNBUFFERED=1

EXPOSE 8020

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8020/ready')" || exit 1

ENTRYPOINT ["python", "embed-pro.py"]
