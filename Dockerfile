FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
RUN apt-get update -y \ 
    && apt-get install $NO_RECS -y python3-dev python3-pip \ 
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
WORKDIR project
COPY flax_image_generators flax_image_generators
COPY setup.py setup.py
RUN python3 -m pip install -e .