# Stage 1: Build stage with build-tools
FROM python:3.12-slim AS builder
RUN apt-get update \
 && apt-get install -y \
    build-essential \
    poppler-utils \
    git \
    cmake \
    pkg-config \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# Filter out llama-cpp-python from requirements
RUN grep -v "llama-cpp-python" requirements.txt > requirements-docker.txt
RUN pip wheel --no-cache-dir --wheel-dir=/wheels -r requirements-docker.txt

# Stage 2: Slim runtime
FROM python:3.12-slim
WORKDIR /app

# Copy wheels & install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# Copy your code
COPY . .

ENV STREAMLIT_SERVER_PORT=8501
EXPOSE 8501
CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0"]