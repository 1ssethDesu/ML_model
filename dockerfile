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

# # Copy application source code
# COPY ./app/ /code/app/

# EXPOSE 8888

# # Default command to run the application
# CMD ["python", "app/main.py", "--port", "8888"]

FROM public.ecr.aws/lambda/python:3.11

RUN yum update -y && \
    yum install -y mesa-libGL glib2-devel && \
    yum clean all && \
    rm -rf /var/cache/yum

# Copy requirements first for better caching
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install -r requirements.txt --target ${LAMBDA_TASK_ROOT} --upgrade --no-cache-dir

# Copy the ONNX model file
COPY app/models/model.onnx ${LAMBDA_TASK_ROOT}/app/models/model.onnx

# Copy app source code
COPY ./app ${LAMBDA_TASK_ROOT}