FROM python:3.11-slim

# libgl1/libglib2.0-0: OpenCV's video/image codecs need these even headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY evals ./evals

RUN uv sync --no-dev

ENTRYPOINT ["uv", "run"]
CMD ["python", "evals/run_eval.py"]
