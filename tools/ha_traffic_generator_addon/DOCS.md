# ESPectre Traffic Generator

The ESPectre Traffic Generator sends a steady stream of packets from Home Assistant to your ESPectre sensors, which they use to measure motion. Its panel also switches devices between their own internal traffic and this external traffic, and shows sensing diagnostics.

## Requirements

- **Home Assistant OS**, 64-bit (`aarch64` or `amd64`), with an administrator account and Internet access. Container, Core, Supervised, and 32-bit installations are not supported.
- **ESPectre 3.0 firmware** (prereleases included) with the ESPHome, Native, or Matter frontend, or an ESPHome device with the `espectre` component, already on Wi-Fi.
- Home Assistant must be able to reach the devices on UDP port **5555**.
- To control devices from the panel, they must be in Home Assistant through **ESPHome** or **Native MQTT Discovery**, with their traffic entities enabled. Matter devices can receive traffic but cannot be controlled from the panel.

## Installation

1. Open **Settings → Apps → Install app**.
2. Select **⋮ → Repositories**, add `https://github.com/francescopace/espectre`, and close the dialog.
3. Search for **ESPectre Traffic Generator** and select **Install**. If it does not appear, select **⋮ → Check for updates**. The first installation can take a few minutes.
4. Open **Configuration** and keep the defaults for the first test, including an empty **Source IPv4 address**.
5. Go back to **Info**, enable **Start on boot** and **Watchdog**, and select **Start**.
6. Select **Open Web UI**, or enable **Show in sidebar** to add **ESPectre Traffic** to the menu.

You do not need Python, Docker, a token, or an open router port: Home Assistant handles access to the panel. Keep protection mode enabled. Only administrators see the sidebar entry.

## Connect your devices

1. Open the web UI. It lists the ESPectre devices already in Home Assistant.
2. Check that the targets, UDP port, and packet rate match your devices. Select **External** on a device, or select up to 32 devices and choose **External** above the table, then confirm.
3. In the table, check traffic, accepted CSI, and occupancy. In Home Assistant or ESPectre Monitor, wait until calibration is done and sensing is ready. Then walk through the room and check the motion entity.

**Internal** switches a device back to its own traffic generator. The panel changes a device only when you ask; it never enables entities or changes rates or targets. External mode survives device restarts. Configure Matter devices in [ESPectre Device settings](https://espectre.dev/tools/device-settings/) instead.

## The panel

What you see for each device:

- **IP address:** from Home Assistant's DHCP records, or from the device's configuration link. It shows where the device was last seen, not that it is reachable now. **—** means unknown.
- **Chip:** shown when Home Assistant knows the hardware model.
- **Wi-Fi RSSI:** signal strength in dBm, if that sensor is enabled in Home Assistant.
- **Generator / s:** packets the device's own generator sent; zero in external mode.
- **TX / s** and **RX / s:** all network packets the device sent and received (UDP, ICMP, TCP, and so on). Older firmware may show a different TX count or omit these values.
- **Occupancy:** how much valid CSI the detector receives. This is not the Matter occupancy sensor.

Missing or disabled values show **—**, never zero. The panel finds a device by its traffic source select, whatever its name. It finds Wi-Fi RSSI by its type: the first diagnostic sensor that reports a signal strength in dBm. It finds the other diagnostics by the name the firmware gives them, so a sensor renamed in ESPHome YAML shows **—**; renaming it in Home Assistant is fine. With the ESPHome `espectre` component, name its sensors Generator Rate, Traffic TX Rate, Traffic RX Rate, CSI Accepted Rate, and CSI Temporal Occupancy. Enable disabled entities in **Settings → Devices & services → Entities**.

The generator summary shows the rate per target and any send errors since the app started. Sent packets do not prove that devices received them. **Stop** pauses the traffic and **Start** resumes it; device modes do not change, so devices in external mode lose their traffic while stopped. Restarting the app resets the counters and starts traffic again.

### How the panel works

- While the panel is open, it asks each device for fresh diagnostics about once per second, by pressing its **Refresh Diagnostics** button. It stops when you close or hide the page; the traffic keeps running. Devices without that button send diagnostics on their own schedule.
- Values are the latest ones Home Assistant has; a request does not guarantee new values.
- The panel reconnects by itself after an error. A failed mode change is never retried automatically.
- The app talks only to Home Assistant, using the token Home Assistant gives it. It does not scan the network, use mDNS, MQTT credentials, or ESPectre's Direct API. Its web server accepts only requests coming through Home Assistant.

## Configuration

**Save and restart the app after changing options.** Each option has help text in Home Assistant, available in English, Italian, French, German, Simplified Chinese, and Japanese (the panel itself is in English). The names below are the keys in the app's YAML editor.

| Option | Default | Use |
| --- | --- | --- |
| `targets` | `239.255.0.1` | Multicast group or a list of device IPv4 addresses. No hostnames, URLs, or broadcast addresses. |
| `port` | `5555` | Destination UDP port; match the devices' external traffic listener. |
| `rate_pps` | `100` | Packets per second **per target**, from `1` to `1000`. Match the devices' `csi_target_pps`. |
| `source_ip` | empty | Leave empty for automatic interface selection. Advanced: an IPv4 address assigned to the Home Assistant host, not an ESPectre device. |
| `multicast_ttl` | `8` | Multicast hop limit, from `1` to `255`. Use `1` for local-only traffic. Does not change unicast TTL. |
| `dscp` | `46` | Traffic class, from `0` to `63`. `46` requests Expedited Forwarding; `0` requests Best Effort. |

The default multicast group reaches every device at once. If multicast does not work, list each device's IP address in **Target addresses** instead, and reserve those addresses in your router. Never send to a device both by multicast and by its IP, or from two generators at once.

### Switches, VLANs, and routed networks

Switches do not reduce TTL; only routers do. Across VLANs, device IP addresses are usually simplest: allow routing and UDP port `5555` between Home Assistant and the devices. Multicast across routers also needs multicast forwarding and IGMP, and a TTL of at least the number of routers plus one. A higher TTL alone does not get past firewalls or isolation.

If traffic never arrives or stops after a few minutes, check multicast filtering on the access point, IGMP snooping, and the VLAN's IGMP querier. Set `source_ip` only if Home Assistant has several network interfaces and picks the wrong one.

### Advanced: DSCP marking

Leave `dscp` at `46` unless your network has a QoS policy to match, or you want to test Best Effort (`0`). Enter the DSCP value, not an IP TOS value. A higher number is not always a higher priority, and network equipment may ignore the marking. See [RFC 8325](https://www.rfc-editor.org/rfc/rfc8325) for how it maps to Wi-Fi.

## Troubleshooting

- **App missing or installation fails:** check the repository branch, host architecture, and **Settings → System → Logs → Supervisor**.
- **App stops immediately:** check its **Log** tab for invalid options. Clear `source_ip` unless you need a specific local interface.
- **Panel cannot connect to Home Assistant:** update or rebuild the app, and check that Home Assistant Core is running. There is no token to enter.
- **Device missing or controls unavailable:** add the device through ESPHome or MQTT and check that its traffic source select is enabled. Matter devices have no traffic controls.
- **Command not confirmed:** check the device's mode in Home Assistant before retrying; a timeout does not mean the change failed. In bulk changes, each failure is reported separately and successful changes are kept.
- **App runs, but no CSI arrives:** confirm external mode and matching rate, port, and multicast group. Try device IP addresses and check network isolation and firewall rules.
- **Packets sent, but no motion:** sent packets are not received packets. Check the device's diagnostics and calibration with [ESPectre troubleshooting](https://github.com/francescopace/espectre/blob/main/docs/TROUBLESHOOTING.md#no-csi-or-insufficient-input). Watchdog only restarts the app if it crashes; it does not check the sensors.

The log shows the settings at startup, then packet counts and send errors every 60 seconds. Updating the app does not update ESPectre firmware or change device settings.

## Standalone use

Without Home Assistant, you can run the same [generator script](https://github.com/francescopace/espectre/blob/main/tools/ha_traffic_generator_addon/espectre_traffic_generator.py) with plain Python. Set `TARGETS`, `PORT`, and `RATE` in the script, then run `python3 tools/ha_traffic_generator_addon/espectre_traffic_generator.py run` from the repository root. It also supports `start`, `status`, and `stop`.
