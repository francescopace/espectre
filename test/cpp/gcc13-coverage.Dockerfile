FROM ubuntu:24.04

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        ca-certificates \
        cmake \
        g++-13 \
        gcc-13 \
        git \
        make \
        python3 \
        python3-pip \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages --no-cache-dir gcovr==8.6

ENV CC=gcc-13 \
    CXX=g++-13 \
    GCOV=gcov-13

WORKDIR /workspace

ENTRYPOINT ["/workspace/test/cpp/run_coverage.sh"]
