FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

# CPU-only torch: pip's default Linux wheel pulls ~15 unused NVIDIA/CUDA
# packages (this app only runs the embedder/reranker on CPU by design),
# bloating both image size and RAM usage on a memory-constrained host.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -e .

COPY download_index.py ./
RUN python download_index.py && rm download_index.py

EXPOSE 8000
CMD ["sh", "-c", "uvicorn pyrag.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
