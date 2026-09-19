# ESPectre Traffic Generator

## Requirements

- Current 64-bit **Home Assistant OS** (`aarch64` or `amd64`), an administrator account, and Internet access for installation and updates. Container, standalone Core, Supervised, and 32-bit installations are not supported.
- **ESPectre 3.0 firmware**, including prereleases, with the ESPHome, Native, or Matter frontend, already connected to Wi-Fi.d.
- Network access from Home Assistant to the devices on UDP port **5555**.

## Installation

1. Open **Settings → Apps → Install app**.
2. Select **⋮ → Repositories**, add `https://github.com/francescopace/espectre`, and close the dialog.
3. Search for **ESPectre Traffic Generator** and select **Install**. If it is missing, select **⋮ → Check for updates**. The first installation can take several minutes.
4. Open **Configuration** and keep the defaults for a first test, including an empty **Source IPv4 address**.
5. Return to **Info**, enable **Start on boot** and **Watchdog**, and select **Start**.

Home Assistant manages the container for you. There is no separate web interface, and you do not need to install Python or Docker manually. Keep protection mode enabled.

## Connect your devices

1. In **Settings → Devices & services → ESPHome**, open each ESPectre device and set **CSI Traffic Ownership** to **external**. For Native or Matter firmware, use [ESPectre Device settings](https://espectre.dev/tools/device-settings/).
2. Select **Refresh Diagnostics** on the ESPHome device, or connect with [ESPectre Monitor](https://espectre.dev/tools/monitor/), and check the CSI input rate and sensing readiness.
3. Let calibration finish, then walk through the sensing area and check the motion entity.

The app does not change device settings automatically. External mode persists across device restarts.

## Configuration

Each field has help text in Home Assistant. **Save and restart the app after changing options.** The names below are the keys used by the YAML editor, not Home Assistant's `configuration.yaml`.

Field names and help text follow your Home Assistant profile language. English, Italian, French, German, Simplified Chinese, and Japanese are available. The Info and Documentation pages remain in English.

| Option | Default | Use |
| --- | --- | --- |
| `targets` | `239.255.0.1` | Multicast group or a list of device IPv4 addresses. No hostnames, URLs, or broadcast addresses. |
| `port` | `5555` | Destination UDP port; match the devices' external traffic listener. |
| `rate_pps` | `100` | Packets per second **per target**, from `1` to `1000`. Match the devices' `csi_target_pps`. |
| `source_ip` | empty | Leave empty for automatic interface selection. Advanced: an IPv4 address assigned to the Home Assistant host, not an ESPectre device. |
| `multicast_ttl` | `8` | Multicast hop limit, from `1` to `255`. Use `1` for local-only traffic. Does not change unicast TTL. |
| `dscp` | `46` | Traffic class, from `0` to `63`. `46` requests Expedited Forwarding; `0` requests Best Effort. |

The default multicast group can serve all devices that have joined `239.255.0.1`. If multicast does not work, replace it in **Target addresses** with each device's IP address and reserve those addresses in your router's DHCP settings. Do not target a device through both multicast and its individual IP, or run another generator for it at the same time.

### Switches, VLANs, and routed networks

Layer 2 switches do not consume TTL. Across VLANs, device IP addresses (unicast) are usually simplest: allow routing and UDP port `5555` between Home Assistant and the devices. Multicast needs multicast forwarding and IGMP support; increasing TTL alone does not enable it or bypass firewall, isolation, or multicast scope rules. Crossing `N` routers requires a TTL of at least `N + 1`.

If delivery fails or stops after a few minutes, check the access point's multicast filtering, IGMP snooping, and the VLAN's IGMP querier. Use `source_ip` only when Home Assistant has multiple interfaces and the automatic choice is unsuitable.

### Advanced: DSCP marking

Leave `dscp` at `46` unless you are matching a network QoS policy or testing Best Effort (`0`). Enter the DSCP codepoint, not an IP TOS value. Larger numbers do not necessarily mean higher priority, and network equipment may ignore or rewrite the marking. See [RFC 8325](https://www.rfc-editor.org/rfc/rfc8325) for Wi-Fi mapping.

## Troubleshooting

- **App missing or installation fails:** check the repository branch, host architecture, and **Settings → System → Logs → Supervisor**.
- **App stops immediately:** check its **Log** tab for invalid options. Clear `source_ip` unless you need a specific local interface.
- **App runs, but no CSI arrives:** confirm external mode and matching rate, port, and multicast group. Try device IP addresses and check network isolation and firewall rules.
- **Packets sent, but no motion:** sent-packet counts do not prove reception. Check device diagnostics and calibration using [ESPectre troubleshooting](https://github.com/francescopace/espectre/blob/main/docs/TROUBLESHOOTING.md#no-csi-or-insufficient-input). Watchdog monitors the process, not reception at the sensors.

The log shows startup settings immediately, packet counters every 60 seconds, and send errors. App updates do not update ESPectre firmware or change device traffic settings.

## Standalone use

Outside Home Assistant, the same standard-library [generator script](https://github.com/francescopace/espectre/blob/main/tools/ha_traffic_generator_addon/espectre_traffic_generator.py) supports `run`, `start`, `status`, and `stop`. Set `TARGETS`, `PORT`, and `RATE` in the script, then run `python3 tools/ha_traffic_generator_addon/espectre_traffic_generator.py run` from the repository root. Home Assistant users should use the app controls instead.
