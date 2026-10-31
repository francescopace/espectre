#!/usr/bin/with-contenv bashio
set -eu

ESPECTRE_INGRESS_PORT="$(bashio::app.ingress_port)"
export ESPECTRE_INGRESS_PORT
exec /opt/venv/bin/python3 -u /panel.py
