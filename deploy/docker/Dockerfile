FROM ess-repos01.wisers.com:8888/pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && conda install -c conda-forge ruamel.yaml

COPY ailab/ /ailab/
VOLUME /ailab
WORKDIR /ailab

EXPOSE 8080
CMD [ "/bin/sh", "start.sh"]
