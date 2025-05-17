FROM python:3.9-slim

WORKDIR /app

# Copy necessary files
COPY app.py /app/
COPY RNNModel.keras /app/
COPY tokenizer.pkl /app/
COPY requirements.txt /app/

# Install dependencies
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

# Expose correct port (optional, but make sure your app binds to it)
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]
