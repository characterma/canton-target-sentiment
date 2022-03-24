ARG PYTHON_VERSION=3.7

FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

RUN apt-get update && apt-get install -y --no-install-recommends curl locales

 
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} conda-build pyyaml numpy ipython&& \
    /opt/conda/bin/conda clean -ya
     
ENV PATH /opt/conda/bin:$PATH
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

 
RUN pip install torch==1.6.0+cu101 -f https://nelsonliu.me/files/pytorch/whl/torch_stable.html



