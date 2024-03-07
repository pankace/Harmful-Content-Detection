FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04
# cuda-10.0-cudnn7-devel-ubuntu16.04
# docker build . -t cuda-10.0-cudnn7-devel-ubuntu16.04
# docker run -it cuda-10.0-cudnn7-devel-ubuntu16.04
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get -y upgrade
