FROM python:3.10-slim-bullseye
ARG KWJA_VERSION
WORKDIR /app
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -q && apt-get install -yq \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip --no-cache-dir \
    && pip3 install --no-cache-dir kwja==${KWJA_VERSION}

# pre-download models
RUN kwja --text 'こんにちは'

CMD /bin/bash
