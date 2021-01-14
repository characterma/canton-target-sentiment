FROM ess-repos01.wisers.com:8888/pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel

COPY requirements.txt ./
RUN pip install --upgrade --ignore-installed pip && pip install --no-cache-dir -r requirements.txt && conda install -c conda-forge ruamel.yaml

COPY ailab/ /ailab/
VOLUME /ailab
WORKDIR /ailab

EXPOSE 8080
CMD [ "/bin/sh", "start.sh"]
