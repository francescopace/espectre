# ESPectre Traffic Generator for Docker

This container sends sensing traffic to ESPectre devices in external mode. Its web UI sets up the traffic, finds the devices on the network, and switches their traffic source. Use it with Home Assistant Container, or without Home Assistant. It runs the same generator as the [Home Assistant app](../ha_traffic_generator_addon/DOCS.md), which is easier on Home Assistant OS.

## Requirements

- A 64-bit Linux host (`amd64` or `arm64`) with Docker, on the same network as the devices.
- Host networking (`--network host`). On Docker's default bridge network, multicast stays inside Docker and never reaches the devices: the web UI shows packets sent but no replies. Only device IP addresses can work there, through Docker's NAT.
- Docker Desktop on macOS also works, with limits. Turn on **Enable host networking** under **Settings > Resources > Network**. **Search** finds no devices there, so add them by address. Devices answer ping, but the web UI shows no replies. UDP markers sent to the multicast group never leave Docker Desktop, so keep ping.
- ESPectre 3.0 or later with the ESPHome, Native, or Matter frontend. Ping to the multicast group needs ESPectre 3.1.0 or later.
- To find and switch devices, the container must reach them on TCP port `62587`, and mDNS must work between them. Devices built with the `espectre` component that comes with ESPHome do not offer this; they can still receive the traffic, but you switch them in Home Assistant.

## Run it

```bash
docker run -d --name espectre-traffic-generator --network host --restart unless-stopped -v espectre-traffic:/data ghcr.io/francescopace/espectre-traffic-generator:latest
```

With Docker Compose, use [compose.yaml](compose.yaml).

On first start the container creates a sign-in token and prints it in its log:

```bash
docker logs espectre-traffic-generator
```

Open `http://<host>:8099` and sign in with the token. The traffic starts with the container: ping to the multicast group `239.255.0.1` at 100 packets per second.

## Use the web UI

- **Settings**: **Save and apply** checks the values, restarts the traffic with them, and resets the counters. The options are the same as the app's; see [configuration](../ha_traffic_generator_addon/DOCS.md#configuration). Unlike the app, the container sends ping by default.
- **Start** and **Stop**: **Stop** pauses the traffic. Device modes stay as they are, so sensors in external mode get no traffic until you start it again. Restarting the container starts the traffic.
- **Status**: packets sent per second and in total, send errors, and, with ping, the replies from the devices. Replies are the easiest way to tell that the devices receive the traffic.

If the traffic cannot be sent, the status shows **Failed** and the reason, for example a source address that the host does not have. Fix the setting, or select **Start** to try again.

## Devices

The container searches the network with mDNS when it starts, every five minutes while the web UI is open, and when you select **Search**. If a device is on another VLAN or mDNS does not reach it, enter its IPv4 address and select **Add**; the container remembers it. **Remove** forgets a device added this way. A saved address stays listed as **Unreachable** while nothing answers there, so you can remove it.

Each row shows:

- **Device**: the name, chip, and address. **Unreachable** means the last reading failed; hover over it for the reason.
- **Generator / s**: packets from the device's own generator. It drops to zero in external mode.
- **TX / s**, **RX / s**, and **CSI / s**: packets the device sends, receives, and accepts for sensing.
- **Occupancy**: how much usable CSI reaches the detector.
- **Wi-Fi RSSI**: signal strength in dBm.
- **Traffic source**: the device's current source. Pick another one and confirm to switch it. The list shows only the sources the device offers; for example, ESP32-C6 has no `wifi_raw`.
- **Sample spacing**: **Half slot** keeps accepted CSI packets at least half a slot apart; **Distinct** needs only distinct timestamps, which can help on busy channels where packets arrive in pairs. The change applies at once, without a confirmation or a new calibration. **—** means the firmware has no such setting, or the device has not answered yet. See [CSI sample spacing](../../docs/ALGORITHMS.md#temporal-admission).

To switch several devices, tick them, pick a source below the table, and select **Apply**. **Previous internal source** gives each device back the internal source it used before external traffic, or ping if the container has restarted since. Each device succeeds or fails on its own. Afterwards only the devices that failed stay selected, so **Apply** retries just those. **External** needs the generator running.

Under an external device, a warning means it does not match the generator: a different `csi_target_pps`, UDP port, or multicast group. Fix it on the device; the container changes only the traffic source.

The container reads the devices every three seconds, and only while someone has the web UI open. It does not use MQTT or Home Assistant.

## With Home Assistant Container

Run the container on the same host as Home Assistant Container, both with host networking. The web UI port `8099` does not clash with Home Assistant's `8123`, and the container never talks to Home Assistant.

ESPectre's ESPHome devices and Native devices with MQTT Discovery also show their traffic source in Home Assistant, so you can switch them in either place. The web UI shows a change made in Home Assistant at its next reading. Matter devices have no traffic source in Home Assistant: switch them in the web UI or in [Monitor](https://espectre.dev/tools/monitor/).

## Environment variables

| Variable | Default | Use |
| --- | --- | --- |
| `ESPECTRE_UI_TOKEN` | generated | Sign-in token, at least 12 characters. When set, it replaces the saved one. |
| `ESPECTRE_WEB_PORT` | `8099` | Web UI port on the host. |
| `ESPECTRE_WEB_HOST` | `0.0.0.0` | Address the web UI listens on. Use `127.0.0.1` to reach it only from the host. |
| `ESPECTRE_DATA_DIR` | `/data` | Folder for the saved settings, devices, and token. |

The data folder holds `settings.json`, `devices.json` with the addresses you added, and `ui_token`. Delete `ui_token` and restart the container to get a new token.

## Security

- The web UI uses plain HTTP. Keep it on your local network, and do not forward its port from the Internet.
- Sign-ins last 30 days. Restarting the container signs everyone out. After five wrong tokens in a minute, sign-in pauses for a minute.
- The container runs as root, which lets it open ping sockets with Docker's default settings.
- Device changes go through ESPectre's Direct HTTP API, which has no password and is meant for a trusted local network. Anyone signed in to the web UI can switch the devices it lists.

## Troubleshooting

- **A device is missing:** check that it runs ESPectre's own ESPHome, Native, or Matter firmware, that the container uses host networking, and that mDNS reaches it. Otherwise add it by its IPv4 address.
- **No sign-in token in the log:** it is printed only when the token is created. Read `ui_token` in the data folder, or set `ESPECTRE_UI_TOKEN`.
- **Failed with an ICMP socket error:** the host does not allow ping sockets. Set **Traffic type** to UDP markers and [open an issue](https://github.com/francescopace/espectre/issues) with the log.
- **Packets sent, but no replies or no CSI on the devices:** check that the devices are in external mode and that the container uses host networking; `docker inspect -f '{{.HostConfig.NetworkMode}}' espectre-traffic-generator` must print `host`. For multicast, VLANs, and firewalls, see [switches, VLANs, and routed networks](../ha_traffic_generator_addon/DOCS.md#switches-vlans-and-routed-networks) and the app's [troubleshooting](../ha_traffic_generator_addon/DOCS.md#troubleshooting).
- **Never run two generators for the same devices:** stop the container before running `./espectre collect` on them, and do not run it next to the Home Assistant app.

## Build the image

From the repository root:

```bash
docker build -f tools/traffic_generator_docker/Dockerfile -t espectre-traffic-generator .
```
