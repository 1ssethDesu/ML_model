# FROM python:3.11-slim

# # Set the working directory
# WORKDIR /code

# # Install system dependencies
# RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install\
#     libgl1\
#     libgl1-mesa-glx \ 
#     libglib2.0-0 -y && \
#     rm -rf /var/lib/apt/lists/*

# # Install the requirements and libraries
# COPY ./requirements.txt /code/requirements.txt
# RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
# RUN pip install --no-cache-dir gdown

# # Copy application source code
# COPY ./app/ /code/app/

# EXPOSE 8000

# # Default command to run the application
# CMD ["python", "app/main.py", "--port", "8000"]

FROM public.ecr.aws/lambda/python:3.11

# Install system dependencies for OpenCV/YOLO
RUN yum update -y && \
    yum install -y mesa-libGL glib2-devel && \
    yum clean all && \
    rm -rf /var/cache/yum

# Copy app and requirements
COPY ./app ${LAMBDA_TASK_ROOT}
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install Python dependencies
RUN pip install -r requirements.txt --target ${LAMBDA_TASK_ROOT} --upgrade --no-cache-dir

COPY app/models/model.onnx ${LAMBDA_TASK_ROOT}/app/models/model.onnx