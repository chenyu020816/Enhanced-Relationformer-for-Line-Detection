FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    git \
    vim \
    build-essential \
    software-properties-common && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get install -f && \
    apt-get upgrade -y

RUN nvcc --version


RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3-pip && \
    python3 -m pip install --upgrade pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get update && \
    apt-get install -y build-essential python3-dev

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN apt-get update && apt-get install -y python3-dev

COPY requirements.txt ./
RUN pip install --ignore-installed -r requirements.txt

CMD ["bash"]

