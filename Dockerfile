FROM python:3.11.9 AS base

# Change this to be your $USER on HPC(e.g. ENV USER dgalgano)
ENV USER=dgalgano

### Set environment variables
# Set langchain variables
ENV LANGCHAIN_API_KEY=
ENV LANGCHAIN_TRACING_V2=true
ENV LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
# Set huggingface variables
ENV HF_TOKEN=
ENV HF_HUB_CACHE=/home/$USER/.cache/huggingface/hub

# Working directory built by image(don't mount over it)
WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
# Copy the rest of the application files into the container
COPY . .
RUN playwright install
RUN playwright install-deps
RUN apt-get install -y vim

ENTRYPOINT python3 grader.py
