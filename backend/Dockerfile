FROM python:3.12-slim

WORKDIR /app

# Install dependencies untuk lzma support
RUN apt-get update && apt-get install -y \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy app
COPY . /app

# expose port
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
