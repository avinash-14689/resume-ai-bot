FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY app.py .
COPY index.html .

# HuggingFace Spaces runs on port 7860
EXPOSE 7860

# Set environment
ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]
