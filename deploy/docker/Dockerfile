FROM ess-repos01.wisers.com:8888/python:3.6-slim-stretch

COPY ailab/src /ailab/src
COPY ailab/config /ailab/config
COPY ailab/models ailab/models 
COPY ailab/setup.py /ailab/setup.py
COPY ailab/start.sh /ailab/start.sh
COPY ailab/requirements.txt /ailab/requirements.txt

VOLUME /ailab
WORKDIR /ailab

RUN pip install --upgrade --ignore-installed pip && pip install -r requirements.txt

EXPOSE 8080
CMD [ "python", "setup.py" , "develop"]
CMD [ "/bin/sh", "start.sh" ]
