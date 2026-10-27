FROM python:3.12-slim
ARG EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/opt/hf \
    YUKI_DATA_DIR=/data YUKI_AUTH_MODE=anon YUKI_EMBEDDING_MODEL=${EMBEDDING_MODEL}
WORKDIR /app
# CPU-only torch keeps the image far smaller than the default CUDA build
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-full.txt .
RUN pip install -r requirements-full.txt
# bake the embedding model into the image so restarts don't re-download it
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('${EMBEDDING_MODEL}')"
COPY yuki_agent ./yuki_agent
COPY run_server.py .
RUN useradd -m app && mkdir -p /data && chown -R app /data /opt/hf
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=8s --start-period=60s CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:%s/api/health'%os.environ.get('PORT','8000'),timeout=6)"
CMD ["sh", "-c", "uvicorn yuki_agent.api:create_app --factory --host 0.0.0.0 --port ${PORT:-8000}"]
