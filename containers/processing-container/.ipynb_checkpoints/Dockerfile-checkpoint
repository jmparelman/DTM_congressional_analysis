# Build an image for the preprocessing of
# congress data for JMParelman project
# https://github.com/jmparelman/dynamic-nmf

# Pull base image
FROM python:3.7-slim-buster

MAINTAINER ASC "mbod@asc.upenn.edu jmparelman@gmail.com"

# install OS modules - especially Python
RUN apt-get -y update && apt-get install -y --no-install-recommends \
        build-essential \
        python-dev \
        python3-pip \
    && pip3 install -U pip


ENV SPACY_VERSION 3.0.5

# install Python packages using pip install
RUN \
    pip3 install -U pandas \
         tqdm \
         gensim \
         sklearn \
         bs4 \
    && pip3 install -U spacy==${SPACY_VERSION} \
    && python3 -m spacy download en_core_web_sm


# setup working directory
RUN mkdir -p /opt/ml/processing/input/speeches /opt/ml/processing/input/descr /opt/ml/processing/output
COPY data/omit_phrases.csv /opt/ml/input/omit_phrases.csv

ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python3"]
