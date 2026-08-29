# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Memory-conscious Wi-Fi bootstrap for the Micro-ESPectre device runtime."""

import gc
import network
import time

import src.config as config


def cleanup_wifi(wlan):
    """Disable CSI and reset the station interface when it is active."""
    if not wlan.active():
        return

    print("Forcing WiFi/CSI cleanup...")
    try:
        wlan.csi_disable()
    except Exception:
        pass
    if wlan.isconnected():
        wlan.disconnect()
    wlan.active(False)
    time.sleep(1)


def print_wifi_status(wlan):
    """Print the connected station's address, protocol, and bandwidth."""
    protocol_names = {
        network.MODE_11B: "b",
        network.MODE_11G: "g",
        network.MODE_11N: "n",
    }
    protocol = wlan.config("protocol")
    modes = [name for bit, name in protocol_names.items() if protocol & bit]
    protocol_label = "802.11" + "/".join(modes) if modes else f"0x{protocol:02x}"
    bandwidth = (
        "HT20"
        if wlan.config("bandwidth") == wlan.BANDWIDTH_20
        else "unknown"
    )
    ip_address = wlan.ifconfig()[0]
    print(
        f"WiFi connected - IP: {ip_address}, Protocol: {protocol_label}, Bandwidth: {bandwidth}"
    )


def _configured_bssid():
    bssid_hex = getattr(config, "WIFI_BSSID", None)
    if not bssid_hex:
        return None
    bssid_clean = bssid_hex.replace(":", "").replace("-", "")
    return bytes.fromhex(bssid_clean) if len(bssid_clean) == 12 else None


def _configure_station_radio(wlan):
    try:
        wlan.config(band_mode=wlan.BAND_MODE_2G_ONLY)
    except Exception:
        pass
    wlan.config(protocol=network.MODE_11B | network.MODE_11G | network.MODE_11N)
    wlan.config(bandwidth=wlan.BANDWIDTH_20)


def _restart_csi_capture(wlan):
    """Rebuild capture and its receive ring through the public CSI API."""
    wlan.csi_disable()
    wlan.csi_enable(
        buffer_size=config.CSI_BUFFER_SIZE,
        max_data_len=getattr(config, "CSI_CAPTURE_MAX_DATA_LEN", 256),
    )


def _connect_station(wlan, timeout_seconds, *, rearm_csi=False):
    bssid = _configured_bssid()
    channel = int(getattr(config, "WIFI_CHANNEL", 0)) if bssid else 0
    wlan.connect(
        config.WIFI_SSID,
        config.WIFI_PASSWORD,
        bssid=bssid,
        channel=channel,
    )
    while not wlan.isconnected() and timeout_seconds > 0:
        time.sleep(1)
        timeout_seconds -= 1
    if not wlan.isconnected():
        return False
    wlan.config(pm=wlan.PM_NONE)
    if rearm_csi:
        _restart_csi_capture(wlan)
    else:
        wlan.csi_enable(
            buffer_size=config.CSI_BUFFER_SIZE,
            max_data_len=getattr(config, "CSI_CAPTURE_MAX_DATA_LEN", 256),
        )
    time.sleep(1)
    return True


def recover_wifi(wlan, timeout_seconds=30, force_reconnect=False):
    """Reconnect a stale station link and rebuild the CSI capture boundary."""
    if wlan.isconnected() and not force_reconnect:
        wlan.config(pm=wlan.PM_NONE)
        _restart_csi_capture(wlan)
        time.sleep(1)
        return True
    attempt_timeout = max(5, int(timeout_seconds) // 3)
    if not force_reconnect and wlan.active():
        _configure_station_radio(wlan)
        if _connect_station(wlan, attempt_timeout, rearm_csi=True):
            return True
        print("[WARN] WiFi reassociation timed out; resetting the station")
    elif force_reconnect:
        print("[WARN] Resetting the WiFi station to recover CSI")
    for attempt in range(2):
        # A station reset is the strong recovery tier. Release the old capture
        # boundary before stopping Wi-Fi so a corrupted ring is not carried into
        # the next association. _connect_station() allocates a fresh ring.
        try:
            wlan.csi_disable()
        except Exception:
            pass
        wlan.active(False)
        time.sleep(1)
        gc.collect()
        wlan.active(True)
        if not wlan.active():
            continue
        _configure_station_radio(wlan)
        if _connect_station(wlan, attempt_timeout, rearm_csi=False):
            return True
        if attempt == 0:
            print("[WARN] WiFi reconnect timed out; resetting the station again")
    return False


def connect_wifi():
    """Connect Wi-Fi and reserve CSI resources before loading the runtime."""
    print("Activating WiFi interface...")
    gc.collect()
    wlan = network.WLAN(network.STA_IF)
    cleanup_wifi(wlan)

    wlan.active(True)
    if not wlan.active():
        raise RuntimeError("WiFi failed to activate")
    time.sleep(2)

    _configure_station_radio(wlan)

    bssid_hex = getattr(config, "WIFI_BSSID", None)
    bssid = _configured_bssid()
    bssid_info = f" (BSSID: {bssid_hex})" if bssid else ""
    print(f"Connecting to WiFi{bssid_info}...")
    if not _connect_station(wlan, 30):
        raise RuntimeError("Connection timeout")

    print_wifi_status(wlan)
    return wlan


def main():
    """Reserve Wi-Fi resources, then load and run the full application."""
    wlan = connect_wifi()
    try:
        from src.runtime_main import main as run_application

        run_application(wlan)
    except BaseException:
        cleanup_wifi(wlan)
        raise
