from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
IDF_FRONTENDS = ("native", "matter", "streamer")


@pytest.mark.parametrize("frontend", IDF_FRONTENDS)
def test_idf_frontend_defaults_optimize_for_size(frontend):
    defaults = (
        REPO_ROOT / "src" / "cpp" / "frontend" / frontend / "app" / "sdkconfig.defaults"
    ).read_text(encoding="utf-8")

    assert "CONFIG_COMPILER_OPTIMIZATION_SIZE=y" in defaults
    assert "CONFIG_COMPILER_OPTIMIZATION_DEBUG=y" not in defaults
    assert "CONFIG_COMPILER_OPTIMIZATION_PERF=y" not in defaults


def test_native_frontend_defaults_disable_unused_nimble_services():
    defaults = (
        REPO_ROOT / "src" / "cpp" / "frontend" / "native" / "app" / "sdkconfig.defaults"
    ).read_text(encoding="utf-8")

    disabled_services = (
        "CONFIG_BT_NIMBLE_PROX_SERVICE",
        "CONFIG_BT_NIMBLE_ANS_SERVICE",
        "CONFIG_BT_NIMBLE_CTS_SERVICE",
        "CONFIG_BT_NIMBLE_HTP_SERVICE",
        "CONFIG_BT_NIMBLE_IPSS_SERVICE",
        "CONFIG_BT_NIMBLE_TPS_SERVICE",
        "CONFIG_BT_NIMBLE_IAS_SERVICE",
        "CONFIG_BT_NIMBLE_LLS_SERVICE",
        "CONFIG_BT_NIMBLE_SPS_SERVICE",
        "CONFIG_BT_NIMBLE_HR_SERVICE",
        "CONFIG_BT_NIMBLE_BAS_SERVICE",
        "CONFIG_BT_NIMBLE_DIS_SERVICE",
    )

    for service in disabled_services:
        assert f"# {service} is not set" in defaults


def test_native_frontend_defaults_disable_unused_mqtt_websocket_transports():
    defaults = (
        REPO_ROOT / "src" / "cpp" / "frontend" / "native" / "app" / "sdkconfig.defaults"
    ).read_text(encoding="utf-8")

    assert "CONFIG_MQTT_TRANSPORT_SSL=y" in defaults
    assert "# CONFIG_MQTT_TRANSPORT_WEBSOCKET is not set" in defaults
    assert "# CONFIG_MQTT_TRANSPORT_WEBSOCKET_SECURE is not set" in defaults


def test_matter_frontend_defaults_do_not_enable_ota_requestor():
    defaults = (
        REPO_ROOT / "src" / "cpp" / "frontend" / "matter" / "app" / "sdkconfig.defaults"
    ).read_text(encoding="utf-8")

    assert "CONFIG_ENABLE_OTA_REQUESTOR=y" not in defaults
