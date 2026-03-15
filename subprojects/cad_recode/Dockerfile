FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

RUN apt-get update \
    && apt-get install -y git libgl1-mesa-glx libegl1-mesa-dev libxrender1 libxext6 libsm6 \
    && rm -rf /var/lib/apt/lists/*

COPY subprojects/cad_recode/requirements.lock.txt /tmp/requirements.lock.txt
RUN pip install --upgrade pip wheel \
    && pip install -r /tmp/requirements.lock.txt

WORKDIR /workspace
