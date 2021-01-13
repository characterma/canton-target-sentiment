FROM ess-repos01.wisers.com:8888/pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

COPY ailab/ /ailab/

VOLUME /ailab
WORKDIR /ailab

ENV CANTON_SA_DIR="/ailab"
RUN pip install --upgrade --ignore-installed pip && pip install -r requirements.txt && conda install -c conda-forge ruamel.yaml && python setup.py develop

EXPOSE 8080
CMD [ "/bin/sh", "start.sh", "sanic", "cuda"]
