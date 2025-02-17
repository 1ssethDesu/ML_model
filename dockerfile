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
RUN pip install --no-cache-dir gdown

# Copy application source code
COPY ./app/ /code/app/

# Default command to run the application
CMD ["python", "app/main.py", "--port", "8000"]