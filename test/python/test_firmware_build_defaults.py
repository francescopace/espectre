"""
ESPectre - Firmware Build Defaults Tests

Tests for firmware build default configuration.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
IDF_FRONTENDS = ("native", "matter", "streamer")


@pytest.mark.parametrize("frontend", IDF_FRONTENDS)
def test_idf_frontend_defaults_optimize_for_size(frontend):
    defaults = (
        REPO_ROOT / "src" / "cpp" / "frontend" / frontend / "app" / "sdkconfig.defaults"
    ).read_text(encoding="utf-8")

    expected_setting = (
        "CONFIG_COMPILER_OPTIMIZATION_PERF=y"
        if frontend == "streamer"
        else "CONFIG_COMPILER_OPTIMIZATION_SIZE=y"
    )
    assert expected_setting in defaults
    assert "CONFIG_COMPILER_OPTIMIZATION_DEBUG=y" not in defaults
    if frontend == "streamer":
        assert "CONFIG_COMPILER_OPTIMIZATION_SIZE=y" not in defaults
    else:
        assert "CONFIG_COMPILER_OPTIMIZATION_PERF=y" not in defaults


def test_native_frontend_defaults_enable_nimble_peripheral_only():
    defaults = (
        REPO_ROOT / "src" / "cpp" / "frontend" / "native" / "app" / "sdkconfig.defaults"
    ).read_text(encoding="utf-8")

    assert "CONFIG_BT_ENABLED=y" in defaults
    assert "CONFIG_BT_NIMBLE_ENABLED=y" in defaults
    assert "CONFIG_BT_NIMBLE_ROLE_PERIPHERAL=y" in defaults
    assert "CONFIG_BT_NIMBLE_ROLE_CENTRAL=n" in defaults

def test_native_frontend_defaults_enable_mqtt_ssl_without_websocket():
    defaults = (
        REPO_ROOT / "src" / "cpp" / "frontend" / "native" / "app" / "sdkconfig.defaults"
    ).read_text(encoding="utf-8")

    assert "CONFIG_MQTT_TRANSPORT_SSL=y" in defaults

def test_matter_frontend_defaults_do_not_enable_ota_requestor():
    defaults = (
        REPO_ROOT / "src" / "cpp" / "frontend" / "matter" / "app" / "sdkconfig.defaults"
    ).read_text(encoding="utf-8")

    assert "CONFIG_ENABLE_OTA_REQUESTOR=y" not in defaults


def test_matter_frontend_uses_persistent_per_device_commissioning_data():
    app_dir = REPO_ROOT / "src" / "cpp" / "frontend" / "matter" / "app"
    defaults = (app_dir / "sdkconfig.defaults").read_text(encoding="utf-8")
    partitions = (app_dir / "partitions.csv").read_text(encoding="utf-8")

    assert "CONFIG_CUSTOM_COMMISSIONABLE_DATA_PROVIDER=y" in defaults
    assert "# CONFIG_ENABLE_TEST_SETUP_PARAMS is not set" in defaults
    assert "matter_factory,data,0x40,0x3F0000, 0x1000," in partitions
