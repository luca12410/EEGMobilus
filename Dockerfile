# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1


RUN apt-get update && apt-get install -y --no-install-recommends \
      tini bluez libbluetooth-dev \
      build-essential gfortran cmake pkg-config \
      libopenblas-dev liblapack-dev \
      ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install -r requirements.txt -v

COPY . /app
COPY ./src /app/src

ENV EEG_UDP_TARGET=127.0.0.1:9999

RUN useradd -m app && chown -R app:app /app
USER app

ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["python3.11","/src/main.py"]
