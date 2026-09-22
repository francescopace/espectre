# ESPectre Traffic Generator

## Requirements

- Current 64-bit **Home Assistant OS** (`aarch64` or `amd64`), an administrator account, and Internet access for installation and updates. Container, standalone Core, Supervised, and 32-bit installations are not supported.
- **ESPectre 3.0 firmware**, including prereleases, with the ESPHome, Native, or Matter frontend, already connected to Wi-Fi.
- Network access from Home Assistant to the devices on UDP port **5555**.
- For device controls in the panel: ESPectre devices already added to Home Assistant through **ESPHome** or **Native MQTT Discovery**, with their traffic controls enabled. Matter exposes occupancy only and cannot be configured through this panel.

## Installation

1. Open **Settings → Apps → Install app**.
2. Select **⋮ → Repositories**, add `https://github.com/francescopace/espectre`, and close the dialog.
3. Search for **ESPectre Traffic Generator** and select **Install**. If it is missing, select **⋮ → Check for updates**. The first installation can take several minutes.
4. Open **Configuration** and keep the defaults for a first test, including an empty **Source IPv4 address**.
5. Return to **Info**, enable **Start on boot** and **Watchdog**, and select **Start**.
6. Select **Open Web UI**, or enable **Show in sidebar** to keep **ESPectre Traffic** in the Home Assistant menu.

Home Assistant manages the container and authenticates access to the panel through Ingress. You do not need to install Python or Docker manually, create a token, or open a web port on your router. Keep protection mode enabled. The sidebar entry is restricted to administrators.

## Connect your devices

1. Open the app's web UI. The panel lists ESPectre devices already in Home Assistant and updates automatically while visible; it does not scan the network or use Direct HTTP.
2. Check that the configured targets, UDP port, and packet rate match your devices. Select **External** on a row, or select up to 32 devices and choose **External** above the table. Confirm to apply. **Internal** restores the device's configured internal generator; it does not delete the device or change its generator type.
3. Check traffic, accepted CSI, and occupancy in the table. The button for the current mode is disabled. Check calibration and sensing readiness in the device's HA entities or ESPectre Monitor, then walk through the sensing area and check the motion entity.

The app changes traffic ownership only after an explicit action; it does not enable entities, alter generator targets, or change device rates. External mode persists across device restarts. Configure Matter devices separately through [ESPectre Device settings](https://espectre.dev/tools/device-settings/); the app can still send them UDP traffic.

## Panel diagnostics and access

The IP address below each device name comes from Home Assistant's DHCP discovery cache, matched by the device's MAC address. Updates arrive through HA's discovery subscription; the add-on does not scan the network or resolve hostnames. If DHCP data is unavailable, the panel uses a literal IP from the device's configuration URL, or shows **—**. Cached addresses are not proof that a device is currently reachable.

The chip appears before the IP when HA provides recognizable hardware metadata. For ESPHome devices whose model is replaced by the project name, the app reads the original model through HA's diagnostics API. Only the chip is cached in memory, for up to an hour; firmware version changes invalidate it on the next inventory read. Unavailable or unsupported diagnostics leave the chip blank without affecting controls. No firmware update or additional permission is required.

**Wi-Fi RSSI** shows the device's reported signal strength in dBm. The corresponding sensor must be exposed and enabled in Home Assistant; missing or unavailable values show **—**.

Use **Stop** beside the generator status to pause UDP traffic, and **Start** to resume it. The panel and diagnostics stay available. These controls do not change device modes: sensors using external traffic will lose that traffic while the generator is stopped. Counters are retained across Start/Stop; restarting the app resets them and starts traffic again. The ESPectre logo opens the project website.

The generator summary shows the configured rate per target and send errors since app startup. These values do not prove reception. On each device, **Generator / s** counts successful internal generator sends and is zero in external mode. **TX / s** and **RX / s** measure all station network packets accepted or delivered by the driver, including UDP, ICMP, and TCP acknowledgements. They exclude radio control frames and raw Wi-Fi injection. These definitions require current firmware; older firmware may omit the new entities and retain the former mixed traffic count in the TX entity. **Occupancy** is valid CSI coverage, not the motion/occupancy binary sensor exposed by Matter.

The panel receives entity changes over WebSocket through the add-on. Visible pages share one persistent subscription to Home Assistant; states and registries are loaded at connection and refreshed when registries change, not polled every second. The add-on also pushes its own generator counters to the page.

ESPectre publishes diagnostics on request, so the add-on still presses each available device's **Refresh Diagnostics** button once per second while at least one page is watching. Requests do not overlap; slow responses reduce the sampling frequency. Hidden pages and open confirmation dialogs disconnect from the stream. When the last page disconnects, the subscription and diagnostic requests stop, but the UDP generator keeps running. Returning to the page or reconnecting to HA loads a fresh snapshot before resuming updates.

Measurements are Home Assistant's latest reported samples. A diagnostic request does not guarantee new values, and devices without an available diagnostic button show whatever HA last received. Missing, disabled, or unavailable measurements show **—**, never a synthetic zero. The panel reconnects automatically after an HA API error; failed mode changes are never retried automatically.

The panel recognizes the standard ESPectre integration identifiers, so renaming devices or entities in Home Assistant is supported. Custom ESPHome YAML that renames the underlying firmware entities may prevent recognition. Missing or ambiguous traffic controls are not operated. Enable disabled entities yourself in **Settings → Devices & services → Entities** if needed.

The app requires Home Assistant Core API access, using the Supervisor-provided token only in its backend. Its web server accepts only the Ingress proxy, and mutations require a request token. No browser token, MQTT credentials, mDNS discovery, or Direct API connection is used for device management.

## Configuration

Each field has help text in Home Assistant. **Save and restart the app after changing options.** The names below are the keys used by the YAML editor, not Home Assistant's `configuration.yaml`.

Field names and help text follow your Home Assistant profile language. English, Italian, French, German, Simplified Chinese, and Japanese are available. The panel, Info, and Documentation pages remain in English.

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
- **Panel cannot connect to Home Assistant:** update or rebuild the app to apply its Core API permission, and check that Home Assistant Core is running. There is no user token to enter.
- **Device missing or controls unavailable:** add the device through ESPHome or MQTT and check that its entities are enabled and available. The list updates automatically. Matter-only devices do not expose traffic controls. Custom firmware entity names may not match the standard identifiers.
- **Command not confirmed:** check the device's mode in Home Assistant before retrying. A timeout does not prove that the device rejected the command; bulk operations report failures individually and do not roll back successful changes.
- **App runs, but no CSI arrives:** confirm external mode and matching rate, port, and multicast group. Try device IP addresses and check network isolation and firewall rules.
- **Packets sent, but no motion:** sent-packet counts do not prove reception. Check device diagnostics and calibration using [ESPectre troubleshooting](https://github.com/francescopace/espectre/blob/main/docs/TROUBLESHOOTING.md#no-csi-or-insufficient-input). Watchdog monitors the process, not reception at the sensors.

The log shows startup settings immediately, packet counters and send-error totals every 60 seconds. App updates do not update ESPectre firmware or change device traffic settings.

## Standalone use

Outside Home Assistant, the same standard-library [generator script](https://github.com/francescopace/espectre/blob/main/tools/ha_traffic_generator_addon/espectre_traffic_generator.py) supports `run`, `start`, `status`, and `stop`. Set `TARGETS`, `PORT`, and `RATE` in the script, then run `python3 tools/ha_traffic_generator_addon/espectre_traffic_generator.py run` from the repository root. Home Assistant users should use the app controls instead.
