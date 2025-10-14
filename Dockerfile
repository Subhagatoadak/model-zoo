# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY bhm_pymc ./bhm_pymc
COPY configs ./configs
COPY example_notebook.ipynb ./example_notebook.ipynb
COPY tests ./tests
COPY mkdocs.yml ./mkdocs.yml
COPY docs ./docs
COPY scripts ./scripts

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "model_zoo.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
