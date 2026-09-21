FROM python:3.12.8-slim-bookworm

ARG LIBRADTRAN_VERSION=2.0.6
ARG REPTRAN_URL="https://www.libradtran.org/lib/exe/fetch.php?media=download:reptran_2024_all.tar.gz"

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    bison \
    build-essential \
    ca-certificates \
    cm-super \
    dvipng \
    flex \
    gawk \
    gfortran \
    ghostscript \
    libgsl-dev \
    libnetcdf-dev \
    lmodern \
    make \
    perl \
    texlive-fonts-extra \
    texlive-fonts-recommended \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-latex-recommended \
    tzdata \
    wget \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

RUN wget -O libRadtran.tar.gz "https://www.libradtran.org/download/libRadtran-${LIBRADTRAN_VERSION}.tar.gz" \
 && wget -O reptran.tar.gz "${REPTRAN_URL}" \
 && tar -xzf libRadtran.tar.gz \
 && rm libRadtran.tar.gz \
 && tar -xzf reptran.tar.gz -C "libRadtran-${LIBRADTRAN_VERSION}" \
 && rm reptran.tar.gz \
 && cd "libRadtran-${LIBRADTRAN_VERSION}" \
 && PYTHON=python3 ./configure \
 && mkdir -p /opt/local \
 && ln -s /usr/include /opt/local/include \
 && make -j"$(nproc)"

COPY container/requirements.txt /tmp/requirements.txt

RUN python -m pip install --no-cache-dir --upgrade \
    pip==24.2 \
    setuptools==72.1.0 \
    wheel==0.44.0 \
 && python -m pip install --no-cache-dir -r /tmp/requirements.txt

ENV PATH="/opt/libRadtran-${LIBRADTRAN_VERSION}/bin:${PATH}" \
    UVSPEC="/opt/libRadtran-${LIBRADTRAN_VERSION}/bin/uvspec" \
    LIBRADTRAN_DATA="/opt/libRadtran-${LIBRADTRAN_VERSION}/data" \
    LIBRADTRAN_DATA_PATH="/opt/libRadtran-${LIBRADTRAN_VERSION}/data" \
    MPLBACKEND="Agg" \
    PYTHONUNBUFFERED="1"

WORKDIR /workspace

CMD ["bash", "scripts/reproduce.sh"]
