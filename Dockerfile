FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HUB_DISABLE_XET=1
WORKDIR /app
# CPU-only torch keeps the image ~1.5 GB smaller
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY vectormatch ./vectormatch
COPY api ./api
COPY ui ./ui
COPY scripts ./scripts
COPY benchmarks ./benchmarks
EXPOSE 8000 8501
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
