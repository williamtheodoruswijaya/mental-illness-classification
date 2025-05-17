FROM python:3.9-slim

WORKDIR /app

COPY . /app/

RUN apt-get update && apt-get install -y \
    build-essential \
    libjpeg-dev \
    liblapack-dev \
    libopenblas-dev \
    libpng-dev \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
