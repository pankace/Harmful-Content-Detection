FROM nvidia/cuda:10.0-cudnn7-devel

ENV DEBIAN_FRONTEND noninteractive
ENV CUDA_HOME /usr/local/cuda