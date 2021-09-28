# Build an image for the preprocessing of
# congress data for JMParelman project
# https://github.com/jmparelman/dynamic-nmf

# Pull base image
FROM python:3.7-slim-buster

MAINTAINER ASC "mbod@asc.upenn.edu jmparelman@gmail.com"

# setup working directory
RUN mkdir -p /data/input /data/output

# install OS modules - especially Python
RUN apt-get -y update && apt-get install -y --no-install-recommends \
        build-essential \
        python-dev \
        python3-pip \
    && pip3 install -U pip


# install Python packages using pip install
RUN \
    pip3 install -U pandas \
         joblib \
         sklearn

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml:${PATH}"

COPY scripts/run_nmf.py /opt/ml/run_nmf.py

ENTRYPOINT ["python3","/opt/ml/run_nmf.py"]
