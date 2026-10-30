# syntax=docker/dockerfile:1
# Build from the repository root:
#   docker build -f tools/traffic_generator_docker/Dockerfile .

# Python 3.14.9 on Alpine.
ARG PYTHON_IMAGE=python:3.14-alpine@sha256:601e99e090cb08c2784a42395487709500bc39930307fe72f3f11a8f3c430762

# Resolve wheels on the build platform, so arm64 images build without emulation.
FROM --platform=$BUILDPLATFORM ${PYTHON_IMAGE} AS dependencies
ARG TARGETARCH
COPY tools/ha_traffic_generator_addon/requirements.txt /requirements.txt
COPY requirements.txt /repository-requirements.txt
# Device discovery uses the CLI's zeroconf pin, so both stay on one version.
RUN grep -E '^zeroconf==' /repository-requirements.txt | cut -d' ' -f1 >> /requirements.txt \
    && case "$TARGETARCH" in \
      amd64) machine=x86_64 ;; \
      arm64) machine=aarch64 ;; \
      *) echo "Unsupported architecture: $TARGETARCH" >&2; exit 1 ;; \
    esac \
    && pip install --no-cache-dir --disable-pip-version-check --target /dependencies \
      --only-binary=:all: --implementation cp --python-version 3.14 \
      --platform "musllinux_1_2_$machine" --platform "musllinux_1_1_$machine" \
      -r /requirements.txt

FROM ${PYTHON_IMAGE}
ARG VERSION=dev
LABEL org.opencontainers.image.title="ESPectre Traffic Generator" \
      org.opencontainers.image.description="Sends external sensing traffic to ESPectre devices, finds them on the network, and switches their traffic source from a web UI." \
      org.opencontainers.image.source="https://github.com/francescopace/espectre" \
      org.opencontainers.image.url="https://espectre.dev/" \
      org.opencontainers.image.licenses="GPL-3.0-only" \
      org.opencontainers.image.version="${VERSION}"

ENV PYTHONPATH=/opt/dependencies:/app/python \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ESPECTRE_DATA_DIR=/data \
    ESPECTRE_WEB_PORT=8099

COPY --from=dependencies /dependencies /opt/dependencies
COPY tools/ha_traffic_generator_addon/espectre_traffic_generator.py tools/ha_traffic_generator_addon/traffic.py tools/ha_traffic_generator_addon/icon.png /app/ha_traffic_generator_addon/
COPY tools/ha_traffic_generator_addon/static/panel.css /app/ha_traffic_generator_addon/static/
COPY src/python/espectre_cli/__init__.py src/python/espectre_cli/device_discovery.py src/python/espectre_cli/device_transport.py /app/python/espectre_cli/
COPY src/python/micro_espectre/__init__.py src/python/micro_espectre/protocol.py /app/python/micro_espectre/
COPY tools/traffic_generator_docker/server.py tools/traffic_generator_docker/devices.py /app/traffic_generator_docker/
COPY tools/traffic_generator_docker/static /app/traffic_generator_docker/static/

VOLUME /data
EXPOSE 8099
STOPSIGNAL SIGTERM
CMD ["python3", "/app/traffic_generator_docker/server.py"]
