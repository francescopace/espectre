from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MATTER_APP = REPO_ROOT / "src" / "cpp" / "frontend" / "matter" / "app"
APP_MAIN = MATTER_APP / "main" / "app_main.cpp"
SDKCONFIG_DEFAULTS = MATTER_APP / "sdkconfig.defaults"


def read(path):
    return path.read_text(encoding="utf-8")


def test_matter_commissioning_window_advertises_ble():
    app_main = read(APP_MAIN)

    assert "OpenBasicCommissioningWindow" in app_main
    assert "CommissioningWindowAdvertisement::kAllSupported" in app_main
    assert "CommissioningWindowAdvertisement::kDnssdOnly" not in app_main
    assert "RendezvousInformationFlag::kBLE" in app_main


def test_matter_does_not_apply_csi_wifi_policy_before_commissioning():
    app_main = read(APP_MAIN)

    assert "StandaloneWifiManager::apply_started_csi_policy" not in app_main
    assert "WIFI_EVENT_STA_START" not in app_main
    assert "set_runtime_services_armed(true)" in app_main
    assert "DeviceEventType::kCommissioningComplete" in app_main


def test_matter_defaults_advertise_occupancy_sensor_identity():
    defaults = read(SDKCONFIG_DEFAULTS)

    assert 'CONFIG_DEVICE_VENDOR_ID=0xFFF1' in defaults
    assert 'CONFIG_DEVICE_PRODUCT_ID=0x8000' in defaults
    assert 'CONFIG_DEVICE_TYPE=0x0107' in defaults
    assert 'CONFIG_ENABLE_COMMISSIONABLE_DEVICE_TYPE=y' in defaults
    assert 'CONFIG_BLE_DEVICE_NAME_PREFIX="ESPectre-"' in defaults
    assert 'CONFIG_ESPECTRE_MATTER_NODE_LABEL="ESPectre Matter"' in defaults


def test_matter_defaults_keep_normal_wifi_for_commissioning():
    defaults = read(SDKCONFIG_DEFAULTS)

    assert "CONFIG_ESP_WIFI_CSI_ENABLED=y" in defaults
    assert "CONFIG_ESP_WIFI_AMPDU_TX_ENABLED=n" not in defaults
    assert "CONFIG_ESP_WIFI_AMPDU_RX_ENABLED=n" not in defaults
    assert "CONFIG_ESP_WIFI_STA_DISCONNECTED_PM_ENABLE=n" not in defaults
