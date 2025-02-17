FROM python:3.11-slim

# Set the working directory
WORKDIR /code

# Install system dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install\
    libgl1\
    libgl1-mesa-glx \ 
    libglib2.0-0 -y && \
    rm -rf /var/lib/apt/lists/*

# Install the requirements and libraries
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy application source code
COPY ./app/ /code/app/

# Create models directory
RUN mkdir -p /code/app/models

# Copy environment file
COPY .env /code/

# Install gdown and download model
RUN pip install --no-cache-dir gdown && \
    set -a && \
    . /code/.env && \
    set +a && \
    if [ ! -f "/code/app/models/model.pt" ]; then \
        echo "Downloading model..."; \
        gdown "$MODEL_URL" -O "/code/app/models/model.pt" --quiet --fuzzy; \
    else \
        echo "Model already exists, skipping download."; \
    fi

# Default command to run the application
CMD ["python", "app/main.py", "--port", "8000"]