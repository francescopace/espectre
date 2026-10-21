# Home Assistant base 2026.10.0 (Alpine 3.24).
FROM ghcr.io/home-assistant/base:latest@sha256:8fde4170393bdf7a121ea4d6e34b417e1d0b90125948459f9f9f70cdf5d77c16

ARG BUILD_VERSION=0.2.0

LABEL \
  io.hass.arch="aarch64|amd64" \
  io.hass.type="app" \
  io.hass.version="${BUILD_VERSION}"

COPY requirements.txt /requirements.txt
RUN apk add --no-cache python3 py3-pip \
    && python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r /requirements.txt

COPY run.sh /run.sh
COPY traffic.py espectre_traffic_generator.py ha_client.py panel.py /
COPY static /static
COPY icon.png /icon.png

RUN chmod a+x /run.sh

CMD ["/run.sh"]
