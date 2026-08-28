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

    try:
        wlan.config(band_mode=wlan.BAND_MODE_2G_ONLY)
    except Exception:
        pass
    wlan.config(protocol=network.MODE_11B | network.MODE_11G | network.MODE_11N)
    wlan.config(bandwidth=wlan.BANDWIDTH_20)

    bssid_hex = getattr(config, "WIFI_BSSID", None)
    bssid = None
    if bssid_hex:
        bssid_clean = bssid_hex.replace(":", "").replace("-", "")
        if len(bssid_clean) == 12:
            bssid = bytes.fromhex(bssid_clean)
    bssid_info = f" (BSSID: {bssid_hex})" if bssid else ""
    print(f"Connecting to WiFi{bssid_info}...")
    wlan.connect(config.WIFI_SSID, config.WIFI_PASSWORD, bssid=bssid)

    timeout = 30
    while not wlan.isconnected() and timeout > 0:
        time.sleep(1)
        timeout -= 1
    if not wlan.isconnected():
        raise RuntimeError("Connection timeout")

    print_wifi_status(wlan)
    wlan.config(pm=wlan.PM_NONE)
    wlan.csi_enable(
        buffer_size=config.CSI_BUFFER_SIZE,
        max_data_len=getattr(config, "CSI_CAPTURE_MAX_DATA_LEN", 256),
    )
    time.sleep(1)
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
