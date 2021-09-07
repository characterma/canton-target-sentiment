FROM ess-repos01.wisers.com:8888/python:3.6-slim-stretch

COPY requirements.txt ./
RUN pip install --upgrade --ignore-installed pip && pip install --no-cache-dir -r requirements.txt --upgrade --ignore-installed && export LC_ALL=C.UTF-8 && export LANG=C.UTF-8 && python -m spacy download en_core_web_sm

COPY ailab/ /ailab/
VOLUME /ailab
WORKDIR /ailab

EXPOSE 8080
CMD [ "/bin/sh", "start.sh"]
