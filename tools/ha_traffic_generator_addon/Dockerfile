FROM ghcr.io/home-assistant/base:latest@sha256:93ef607824e3f27e868f11b10938283a98bf880ed57bcf8eaa81c6c2d521f6f5

ARG BUILD_VERSION=0.1.0

LABEL \
  io.hass.arch="aarch64|amd64" \
  io.hass.type="app" \
  io.hass.version="${BUILD_VERSION}"

RUN apk add --no-cache python3

COPY run.sh /run.sh
COPY traffic.py espectre_traffic_generator.py /

RUN chmod a+x /run.sh

CMD ["/run.sh"]
