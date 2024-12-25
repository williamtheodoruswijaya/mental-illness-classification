FROM python:3.9-slim

WORKDIR /app

COPY app.py /app/
COPY RNNModel.keras /app/
COPY tokenizer.pkl /app/
COPY requirements.txt .
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
COPY . .

EXPOSE 5000

CMD ["python", "app.py"]