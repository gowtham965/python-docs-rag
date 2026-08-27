FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .

COPY download_index.py ./
RUN python download_index.py && rm download_index.py

EXPOSE 8000
CMD ["sh", "-c", "uvicorn pyrag.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
