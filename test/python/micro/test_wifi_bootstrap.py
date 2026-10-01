# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Micro-ESPectre Wi-Fi bootstrap: association, recovery, and fail-closed cleanup."""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[3] / "src/python/micro_espectre"


class FakeWlan:
    """Station interface that associates after a configurable number of seconds."""

    BAND_MODE_AUTO = 0
    BAND_MODE_2G_ONLY = 1
    BANDWIDTH_20 = 20
    PM_NONE = 0

    def __init__(self, *, active=True, connected=False, join_after=None, protocol=7, bandwidth=20):
        self._active = active
        self._connected = connected
        self.join_after = join_after
        self.protocol = protocol
        self.bandwidth = bandwidth
        self.calls = []
        self.clock = None
        self.reject_band_mode = False
        self.activation_fails = False

    def active(self, value=None):
        if value is None:
            return self._active
        self.calls.append(("active", value))
        self._active = value and not self.activation_fails

    def isconnected(self):
        if self._connected:
            return True
        if self.join_after is not None and self.clock is not None and self.clock.elapsed >= self.join_after:
            self._connected = True
        return self._connected

    def connect(self, ssid, password, bssid=None, channel=0):
        self.calls.append(("connect", ssid, bssid, channel))

    def disconnect(self):
        self.calls.append(("disconnect",))
        self._connected = False

    def csi_enable(self, **kwargs):
        self.calls.append(("csi_enable", kwargs))

    def csi_disable(self):
        self.calls.append(("csi_disable",))

    def config(self, *args, **kwargs):
        if args:
            if args[0] == "protocol":
                if self.protocol is None:
                    raise OSError("unsupported")
                return self.protocol
            if args[0] == "bandwidth":
                if self.bandwidth is None:
                    raise OSError("unsupported")
                return self.bandwidth
            raise KeyError(args[0])
        if "band_mode" in kwargs and self.reject_band_mode:
            raise OSError("band mode unsupported")
        self.calls.append(("config", kwargs))

    def ifconfig(self):
        return ("192.168.1.50", "255.255.255.0", "192.168.1.1", "192.168.1.1")

    def names(self):
        return [call[0] for call in self.calls]


class Clock:
    def __init__(self):
        self.elapsed = 0

    def sleep(self, seconds):
        self.elapsed += seconds


@pytest.fixture
def bootstrap(monkeypatch):
    package = ModuleType("src")
    package.__path__ = [str(ROOT)]
    monkeypatch.setitem(sys.modules, "src", package)
    config = importlib.import_module("config")
    monkeypatch.setattr(config, "WIFI_SSID", "lab", raising=False)
    monkeypatch.setattr(config, "WIFI_PASSWORD", "secret", raising=False)
    monkeypatch.setattr(config, "WIFI_BSSID", "", raising=False)
    monkeypatch.setattr(config, "WIFI_CHANNEL", 6, raising=False)
    monkeypatch.setattr(config, "CSI_BUFFER_SIZE", 8, raising=False)
    monkeypatch.setitem(sys.modules, "src.config", config)
    package.config = config
    console = importlib.import_module("console_output")
    monkeypatch.setitem(sys.modules, "src.console_output", console)
    network = SimpleNamespace(
        MODE_11B=1, MODE_11G=2, MODE_11N=4, STA_IF=0, WLAN=Mock(),
    )
    monkeypatch.setitem(sys.modules, "network", network)
    native = SimpleNamespace(prepare_tx_rate=Mock(), apply_tx_rate=Mock())
    monkeypatch.setitem(sys.modules, "espectre_native_wifi", native)
    spec = importlib.util.spec_from_file_location("wifi_bootstrap_under_test", ROOT / "wifi_bootstrap.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    clock = Clock()
    monkeypatch.setattr(module, "time", SimpleNamespace(sleep=clock.sleep))
    monkeypatch.setattr(module, "gc", SimpleNamespace(collect=lambda: None))
    monkeypatch.setattr(module, "print_log", Mock())
    module.clock = clock
    module.network_stub = network
    module.native = native
    module.test_config = config
    return module


def wlan_with(bootstrap, **kwargs):
    wlan = FakeWlan(**kwargs)
    wlan.clock = bootstrap.clock
    return wlan


def test_cleanup_leaves_an_inactive_station_alone(bootstrap):
    wlan = wlan_with(bootstrap, active=False)

    bootstrap.cleanup_wifi(wlan)

    assert wlan.calls == []


def test_cleanup_disables_csi_disconnects_and_deactivates(bootstrap):
    wlan = wlan_with(bootstrap, connected=True)

    bootstrap.cleanup_wifi(wlan)

    assert wlan.names() == ["csi_disable", "disconnect", "active"]
    assert wlan.active() is False


def test_cleanup_survives_a_failing_csi_disable(bootstrap):
    wlan = wlan_with(bootstrap)
    wlan.csi_disable = Mock(side_effect=OSError("csi down"))

    bootstrap.cleanup_wifi(wlan)

    assert wlan.active() is False


@pytest.mark.parametrize("protocol, bandwidth, expected", [
    (1 | 2 | 4, 20, "Protocol: 802.11b/g/n, Bandwidth: HT20"),
    (4, 40, "Protocol: 802.11n, Bandwidth: unknown"),
    (0x40, 20, "Protocol: 0x40, Bandwidth: HT20"),
    (None, None, "Protocol: unknown, Bandwidth: unknown"),
])
def test_status_line_reports_protocol_and_bandwidth(bootstrap, protocol, bandwidth, expected):
    wlan = wlan_with(bootstrap, protocol=protocol, bandwidth=bandwidth)

    bootstrap.print_wifi_status(wlan)

    level, message = bootstrap.print_log.call_args.args
    assert level == "INFO"
    assert "IP: 192.168.1.50" in message
    assert expected in message


@pytest.mark.parametrize("configured, expected", [
    ("", None),
    (None, None),
    ("aa:bb:cc:dd:ee:ff", bytes.fromhex("aabbccddeeff")),
    ("AA-BB-CC-DD-EE-FF", bytes.fromhex("aabbccddeeff")),
    ("aa:bb:cc", None),
])
def test_configured_bssid_accepts_only_a_full_address(bootstrap, monkeypatch, configured, expected):
    monkeypatch.setattr(bootstrap.test_config, "WIFI_BSSID", configured, raising=False)

    assert bootstrap._configured_bssid() == expected


def test_station_radio_prefers_automatic_band_selection(bootstrap):
    wlan = wlan_with(bootstrap)

    bootstrap._configure_station_radio(wlan)

    assert wlan.calls == [("config", {"band_mode": FakeWlan.BAND_MODE_AUTO})]


def test_station_radio_pins_2g_ht20_when_automatic_band_is_rejected(bootstrap):
    wlan = wlan_with(bootstrap)
    wlan.reject_band_mode = True

    bootstrap._configure_station_radio(wlan)

    assert ("config", {"protocol": 7}) in wlan.calls
    assert ("config", {"bandwidth": FakeWlan.BANDWIDTH_20}) in wlan.calls


def test_station_radio_pins_2g_ht20_without_an_auto_band_constant(bootstrap):
    wlan = wlan_with(bootstrap)
    del FakeWlan.BAND_MODE_AUTO
    try:
        bootstrap._configure_station_radio(wlan)
    finally:
        FakeWlan.BAND_MODE_AUTO = 0

    assert ("config", {"band_mode": FakeWlan.BAND_MODE_2G_ONLY}) in wlan.calls


def test_connect_pins_the_configured_access_point_and_arms_csi(bootstrap, monkeypatch):
    monkeypatch.setattr(bootstrap.test_config, "WIFI_BSSID", "aa:bb:cc:dd:ee:ff", raising=False)
    wlan = wlan_with(bootstrap, join_after=2)

    assert bootstrap._connect_station(wlan, 10) is True

    assert ("connect", "lab", bytes.fromhex("aabbccddeeff"), 6) in wlan.calls
    assert wlan.names().count("csi_enable") == 1
    assert "csi_disable" not in wlan.names()
    bootstrap.native.prepare_tx_rate.assert_called_once()
    bootstrap.native.apply_tx_rate.assert_called_once()


def test_connect_scans_all_channels_without_a_pinned_bssid(bootstrap):
    wlan = wlan_with(bootstrap, join_after=1)

    bootstrap._connect_station(wlan, 10)

    assert ("connect", "lab", None, 0) in wlan.calls


def test_connect_rearms_an_existing_capture_when_requested(bootstrap):
    wlan = wlan_with(bootstrap, join_after=1)

    assert bootstrap._connect_station(wlan, 10, rearm_csi=True) is True

    assert wlan.names()[-2:] == ["csi_disable", "csi_enable"]


def test_connect_gives_up_after_the_timeout_without_arming_csi(bootstrap):
    wlan = wlan_with(bootstrap)

    assert bootstrap._connect_station(wlan, 3) is False

    assert bootstrap.clock.elapsed == 3
    assert "csi_enable" not in wlan.names()
    bootstrap.native.apply_tx_rate.assert_not_called()


def test_recovery_rebuilds_capture_on_a_still_associated_station(bootstrap):
    wlan = wlan_with(bootstrap, connected=True)

    assert bootstrap.recover_wifi(wlan) is True

    assert wlan.names() == ["config", "csi_disable", "csi_enable"]
    assert "connect" not in wlan.names()


def test_recovery_reassociates_an_active_station_before_resetting_it(bootstrap):
    wlan = wlan_with(bootstrap, join_after=2)

    assert bootstrap.recover_wifi(wlan, timeout_seconds=30) is True

    assert ("active", False) not in wlan.calls
    assert "connect" in wlan.names()


def test_recovery_resets_the_station_when_reassociation_times_out(bootstrap):
    wlan = wlan_with(bootstrap, join_after=25)

    assert bootstrap.recover_wifi(wlan, timeout_seconds=30) is True

    assert wlan.calls.index(("active", False)) < wlan.calls.index(("active", True))
    assert bootstrap.print_log.call_args_list[0].args[0] == "WARN"


def test_forced_recovery_always_resets_the_station(bootstrap):
    wlan = wlan_with(bootstrap, connected=True, join_after=0)

    assert bootstrap.recover_wifi(wlan, force_reconnect=True) is True

    assert ("active", False) in wlan.calls
    assert "Resetting the WiFi station to recover CSI" in bootstrap.print_log.call_args_list[0].args[1]


def test_recovery_reports_failure_after_two_station_resets(bootstrap):
    wlan = wlan_with(bootstrap, active=False)

    assert bootstrap.recover_wifi(wlan, timeout_seconds=9) is False

    assert wlan.calls.count(("active", False)) == 2
    assert wlan.calls.count(("active", True)) == 2


def test_recovery_skips_an_attempt_when_the_station_will_not_activate(bootstrap):
    wlan = wlan_with(bootstrap, active=False)
    wlan.activation_fails = True

    assert bootstrap.recover_wifi(wlan, timeout_seconds=9) is False

    assert "connect" not in wlan.names()


def test_recovery_tolerates_a_failing_csi_disable_during_reset(bootstrap):
    wlan = wlan_with(bootstrap, active=False, join_after=1)
    wlan.csi_disable = Mock(side_effect=OSError("csi down"))

    assert bootstrap.recover_wifi(wlan, timeout_seconds=9) is True


def test_connect_wifi_returns_a_connected_station(bootstrap):
    wlan = wlan_with(bootstrap, active=False, join_after=1)
    bootstrap.network_stub.WLAN.return_value = wlan

    assert bootstrap.connect_wifi() is wlan

    assert wlan.isconnected()
    assert wlan.names().count("csi_enable") == 1


def test_connect_wifi_rejects_an_interface_that_will_not_activate(bootstrap):
    wlan = wlan_with(bootstrap, active=False)
    wlan.activation_fails = True
    bootstrap.network_stub.WLAN.return_value = wlan

    with pytest.raises(RuntimeError, match="failed to activate"):
        bootstrap.connect_wifi()


def test_connect_wifi_times_out_without_an_access_point(bootstrap):
    wlan = wlan_with(bootstrap, active=False)
    bootstrap.network_stub.WLAN.return_value = wlan

    with pytest.raises(RuntimeError, match="Connection timeout"):
        bootstrap.connect_wifi()


def test_connect_wifi_mentions_a_pinned_bssid(bootstrap, monkeypatch):
    monkeypatch.setattr(bootstrap.test_config, "WIFI_BSSID", "aa:bb:cc:dd:ee:ff", raising=False)
    wlan = wlan_with(bootstrap, active=False, join_after=1)
    bootstrap.network_stub.WLAN.return_value = wlan

    bootstrap.connect_wifi()

    messages = [call.args[1] for call in bootstrap.print_log.call_args_list]
    assert any("BSSID: aa:bb:cc:dd:ee:ff" in message for message in messages)


def test_main_releases_wifi_when_the_application_fails(bootstrap, monkeypatch):
    wlan = wlan_with(bootstrap, connected=True)
    monkeypatch.setattr(bootstrap, "connect_wifi", lambda: wlan)
    runtime = SimpleNamespace(main=Mock(side_effect=KeyboardInterrupt))
    monkeypatch.setitem(sys.modules, "src.runtime_main", runtime)

    with pytest.raises(KeyboardInterrupt):
        bootstrap.main()

    assert wlan.active() is False


def test_main_hands_the_connected_station_to_the_application(bootstrap, monkeypatch):
    wlan = wlan_with(bootstrap, connected=True)
    monkeypatch.setattr(bootstrap, "connect_wifi", lambda: wlan)
    runtime = SimpleNamespace(main=Mock())
    monkeypatch.setitem(sys.modules, "src.runtime_main", runtime)

    bootstrap.main()

    runtime.main.assert_called_once_with(wlan)
    assert wlan.active() is True
