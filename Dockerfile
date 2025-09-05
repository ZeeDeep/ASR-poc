FROM python:3.12-slim AS base

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . /app

# Expose API port
EXPOSE 8080

# Run API
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8080"]
#CMD ["python", "-m", "src.main"]
