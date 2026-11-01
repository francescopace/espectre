# ESPectre Traffic Generator

ESPectre sensors measure motion from the Wi-Fi packets they receive. This app sends those packets from Home Assistant at a steady rate. It also adds a panel where you pick each sensor's traffic source, its own packets (internal) or the app's traffic (external), and follow its diagnostics.

## Requirements

- Home Assistant OS, 64-bit (`aarch64` or `amd64`), with an administrator account and Internet access. Container, Core, Supervised, and 32-bit installations are not supported; on Home Assistant Container, use the [Docker image](https://github.com/francescopace/espectre/blob/main/tools/traffic_generator_docker/README.md) instead.
- ESPectre 3.0 or later (prereleases included) with the ESPHome, Native, or Matter frontend, or an ESPHome device with the `espectre` component. The device must already be on Wi-Fi.
- Home Assistant must reach the devices by ping, or on UDP port 5555 if you use UDP. Pinging the multicast group needs ESPectre 3.1.0 or later. Older firmware only answers pings sent to its own IP address.
- To control a device from the panel, add it to Home Assistant through ESPHome or Native MQTT Discovery and keep its traffic entities enabled. Matter devices can receive the traffic, but the panel cannot control them.

## Installation

1. Go to **Settings → Apps → Install app**.
2. Open **⋮ → Repositories**, add `https://github.com/francescopace/espectre`, and close the dialog.
3. Find **ESPectre Traffic Generator** and select **Install**. If it is not listed, select **⋮ → Check for updates**. The first installation takes a few minutes.
4. In **Configuration**, set **Traffic type** to `icmp`. On ESPectre 3.1.0 or later, leave the multicast group in **Target addresses**. On older firmware, replace it with the IP address of each device. Leave the other options as they are.
5. Back on **Info**, turn on **Start on boot** and **Watchdog**, then select **Start**.
6. Select **Open Web UI**. Turn on **Show in sidebar** if you want an **ESPectre Traffic** menu entry.

Home Assistant handles access to the panel, so there is no token to create and no router port to open. Leave protection mode on. Only administrators see the sidebar entry.

## Connect your devices

1. Open the web UI. It lists the ESPectre devices that Home Assistant already knows.
2. Check the generator summary at the top. Its rate must match the devices' `csi_target_pps`.
3. Set a device's **Traffic source** to **External** and confirm. To switch several devices at once, tick up to 32 of them, pick a source below the table, and select **Apply**.
4. Watch the TX, CSI, and occupancy columns fill in. Wait until Home Assistant or [Live motion](https://espectre.dev/tools/monitor/) shows that calibration is done, then walk through the room and check the motion entity.

**Traffic source** lists every source the device offers: ping, DNS, DNS over TCP, Wi-Fi raw where the chip supports it, and external. Below the table, **Previous internal source** gives each selected device back the internal source it used before you switched it. If the app has restarted since then, or the mode was changed somewhere else, it falls back to ping. The chosen source survives device restarts.

The panel changes a device only when you ask. It never enables entities or changes rates and targets.

For a Matter device, open [Live motion](https://espectre.dev/tools/monitor/), expand **Advanced sensing settings**, and set **Wi-Fi traffic source** to **External device**.

## The panel

Each row shows:

- **Device**: the name, with the chip and the IP address below it. The IP address comes from Home Assistant's DHCP records or the device's configuration link, so it is where the device was last seen. **Unavailable** in its place means the panel cannot control the device. Hover over it to see why.
- **Generator / s**: packets sent by the device's own generator. It drops to zero in external mode.
- **TX / s** and **RX / s**: all packets the device sends and receives, whatever the protocol.
- **CSI / s**: CSI packets the detector accepted.
- **Occupancy**: how much usable CSI reaches the detector. This is not the Matter occupancy sensor.
- **Wi-Fi RSSI**: signal strength in dBm.
- **Traffic source**: the source the device uses now. Pick another one and confirm to switch it.
- **Packet spacing**: **Half slot** keeps accepted CSI packets at least half a slot apart; **Distinct timestamps** needs only distinct timestamps, which can help on busy channels where packets arrive in pairs. The change applies at once, without a confirmation or a new calibration. **—** means the device does not expose the CSI Packet Spacing select, or it is disabled or unavailable. See [temporal admission](https://github.com/francescopace/espectre/blob/main/docs/ALGORITHMS.md#temporal-admission).

**—** means the value is missing, disabled, or unknown. You can enable hidden entities in **Settings → Devices & services → Entities**.

A device shows up as long as its traffic source select is enabled, whatever it is called. Diagnostics are matched by the names the firmware gives them: renaming a sensor in Home Assistant is fine, but renaming it in ESPHome YAML breaks the match. With the ESPHome `espectre` component, keep these names: Generator Rate, Traffic TX Rate, Traffic RX Rate, CSI Accepted Rate, and CSI Temporal Occupancy. RSSI is matched by type, so any diagnostic signal strength sensor in dBm works.

The summary at the top shows the targets, the rate, and the send errors since the app started. With `icmp` it also counts replies, which is the easiest way to tell that the devices actually receive the traffic.

**Stop** pauses the traffic and **Start** resumes it. Device modes stay as they are, so sensors in external mode get no traffic while the generator is stopped, and **External** cannot be picked until you start it again. Restarting the app resets the counters and starts the traffic.

### What the panel does in the background

While the panel is open and visible, it presses each device's **Refresh Diagnostics** button about once a second to keep the numbers fresh. It stops when nobody has the panel open. The traffic keeps going either way. Devices without that button update on their own schedule.

If the connection to Home Assistant drops, the panel reconnects by itself. It never retries a mode change on its own.

The app talks only to Home Assistant, with the token Home Assistant gives it. It does not scan the network or use mDNS, MQTT, or ESPectre's Direct API. Its web server only accepts requests that come through Home Assistant.

## Configuration

Save and restart the app after changing options. Home Assistant shows help text for each option. The names below are the keys in the YAML editor.

| Option | Default | Use |
| --- | --- | --- |
| `targets` | `239.255.0.1` | The multicast group, or a list of device IPv4 addresses. Hostnames, URLs, and broadcast addresses do not work. |
| `protocol` | `udp` | `icmp` sends pings that the devices answer. `udp` sends markers that get no reply. See [ping or UDP](#ping-or-udp). |
| `port` | `5555` | UDP port on the devices. Used only with `udp`. |
| `rate_pps` | `100` | Packets per second for each target, from `1` to `1000`. Match the devices' `csi_target_pps`. |
| `source_ip` | empty | Leave empty. Set it to one of the Home Assistant host's own IPv4 addresses only to force a network interface. |
| `multicast_ttl` | `8` | How many routers multicast can cross, from `1` to `255`. Use `1` to keep it on the local network. |
| `dscp` | `46` | Traffic class, from `0` to `63`. See [DSCP marking](#advanced-dscp-marking). |

One multicast stream reaches every device. If multicast does not get through, list the device IP addresses instead and reserve them in your router. Send to each device one way only: never by multicast and by IP address at once, and never from two generators. `./espectre collect` runs its own generator, so stop the app before using it on the same devices.

### Ping or UDP

Use `icmp` unless you have a reason not to. Each device answers every ping, and the answer keeps it transmitting while it measures. An empty room then stays quiet.

With `udp`, the devices receive markers and send almost nothing back. The access point will likely change how it transmits to a client that stays silent, and that alone can show up as motion in an empty room. The device's regular updates to Home Assistant are too sparse to help. Pick `udp` when airtime is tight and the devices already send steady traffic of their own, or when you need the multicast group on firmware older than 3.1.0.

Ping uses roughly twice the airtime of UDP. The [airtime estimates](https://github.com/francescopace/espectre/blob/main/docs/CSI.md#network-impact-and-airtime) have the numbers.

### Switches, VLANs, and routed networks

Switches do not count against the TTL. Only routers do. Across VLANs, device IP addresses are usually the simplest option: allow routing and ICMP Echo, or UDP port `5555`, between Home Assistant and the devices. Multicast across routers also needs multicast forwarding, IGMP, and a TTL of at least the number of routers plus one. A higher TTL will not get traffic past a firewall or client isolation.

If traffic never arrives, or stops after a few minutes, look at multicast filtering on the access point, IGMP snooping, and the VLAN's IGMP querier. If Home Assistant has more than one network interface and sends from the wrong one, set `source_ip`.

### Advanced: DSCP marking

Leave `dscp` at `46` (Expedited Forwarding) unless your network has a QoS policy to match, or you want to try Best Effort (`0`). Enter a DSCP value, not an IP TOS byte. A bigger number is not always a higher priority, and plenty of equipment ignores the marking anyway. [RFC 8325](https://www.rfc-editor.org/rfc/rfc8325) explains how it maps to Wi-Fi.

## Troubleshooting

- **The app is not listed or fails to install:** check the host architecture and the Supervisor log in **Settings → System → Logs**.
- **The app stops right away:** its **Log** tab names the invalid option. If `source_ip` is set, try clearing it.
- **The panel cannot reach Home Assistant:** make sure Home Assistant Core is running, then update or rebuild the app.
- **The panel says the session expired:** reload the page.
- **A device is missing or its buttons are disabled:** make sure it comes from ESPHome or MQTT and its traffic source select is enabled. Hover over **Unavailable** for the reason. Matter devices have no controls in the panel.
- **A command is not confirmed:** the change may still go through, and the warning clears by itself when the device confirms it. In a bulk change, each device succeeds or fails on its own. Afterwards only the devices that failed or have not confirmed stay selected, so **Apply** retries just those.
- **The app runs but no CSI arrives:** check that the device is in external mode and that the rate, port, and multicast group match. With `icmp` on the multicast group, check the [firmware version](#requirements). Otherwise, try device IP addresses and look for client isolation or firewall rules.
- **The app stops with an ICMP socket error:** the host does not allow ping sockets. Switch **Traffic type** to `udp` and [open an issue](https://github.com/francescopace/espectre/issues) with the log.
- **Packets go out, but no motion is detected:** with `icmp`, check that the reply count goes up. Then check the device's diagnostics and calibration in the [troubleshooting guide](https://espectre.dev/guides/troubleshooting/#trouble-diagnostics). Watchdog restarts the app if it crashes, but it does not watch the sensors.

The log shows the settings at startup, every **Start** and **Stop** from the panel, and new send errors once a minute. Updating the app does not change the ESPectre firmware or device settings.

## Standalone use

To run the generator without Home Assistant OS, use the [Docker image](https://github.com/francescopace/espectre/blob/main/tools/traffic_generator_docker/README.md). Its web UI sets these options, finds devices on the network, and switches their traffic source through ESPectre's Direct HTTP API.

You can also run the same [generator script](https://github.com/francescopace/espectre/blob/main/tools/ha_traffic_generator_addon/espectre_traffic_generator.py) without Home Assistant, using plain Python. Set `TARGETS`, `PROTOCOL`, `PORT`, `RATE`, `MULTICAST_TTL`, and `DSCP` in its configuration block, then run it from the repository root:

```bash
python3 tools/ha_traffic_generator_addon/espectre_traffic_generator.py run
```

Use `start`, `status`, and `stop` to run it in the background instead. Ping needs no special privileges on macOS. On Linux it needs root, or a group allowed by `net.ipv4.ping_group_range`. On Windows only UDP works.
