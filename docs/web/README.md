# Website

Static single-page app published at `espectre.dev` through GitHub Pages.

## Run locally

```bash
python -m http.server 8090 --directory docs/web
```

Then open `http://localhost:8090`. The Flash tool and the Matter QR reader
need a Chromium-based browser; the BLE connection additionally needs
`localhost` or HTTPS.

## BLE client API

`espectre-ble.js` is a dependency-free client for the ESPectre BLE surface
defined in `docs/ESPECTRE_PROTOCOL.md`. It exposes two globals:
`ESPectreBleClient` and `ESPectreValidationError`. Web Bluetooth needs a
Chromium-based browser and a secure context (HTTPS or `localhost`); check
`ESPectreBleClient.supported` before connecting.

Unlike the rest of the site, the client is **Apache-2.0** licensed (see
`LICENSING.md`), so any web application, including proprietary ones, can
embed it.

```js
const client = new ESPectreBleClient();
client.on('telemetry', ({ movement, threshold, motionState }) => { /* ... */ });
client.on('sysinfo', (values) => console.log(values.chip, values.detector));
client.on('disconnect', () => console.log('device dropped'));

await client.connect();          // opens the browser device chooser
await client.requestSysinfo();   // resolves into a `sysinfo` event
await client.setThreshold(0.35);
await client.disconnect();
```

### Events

Subscribe with `on(event, handler)`, which returns an unsubscribe function;
`off(event, handler)` also works. A throwing handler is logged and never
breaks the client or other handlers.

| Event | Payload | When |
|---|---|---|
| `telemetry` | `{ movement, threshold, motionState }` | Every valid notification; `motionState` is `null` on firmware that omits it |
| `invalid-telemetry` | `byteLength` | A notification failed to parse |
| `sysinfo` | `(values, entries)` — object plus ordered pairs | A snapshot completed (`END` received) |
| `sysinfo-line` | raw line | Every sysinfo line, including `END` |
| `disconnect` | — | Unexpected GATT drop; never fired by `disconnect()` |

### Connection

| Member | Notes |
|---|---|
| `connect({ telemetry, sysinfo })` | Both flags default to `true`; reentrant (returns the in-flight promise or the connected device) |
| `disconnect()` | Idempotent; stops notifications and closes GATT |
| `setTelemetryNotifications(bool)` / `setSysinfoNotifications(bool)` | Toggle streams without disconnecting |
| `connected`, `name`, `device` | Read-only state |

### Commands

Every `set*` method validates locally, throws `ESPectreValidationError` on a
bad argument, and writes the command over the control characteristic. The
matching static `build*Command` functions are pure and return the wire
string, so arguments can be validated without a connected device.

| Method | Command | Validation |
|---|---|---|
| `setThreshold(value)` | `SET_THRESHOLD` | number in `0.0-1.0` |
| `setDetector(name)` | `SET_DETECTOR` | `classic` or `ml` |
| `setWifiConfig({ ssid, password, bssid, channel })` | `SET_WIFI_CONFIG` | `ssid` required; `channel` 0-14 (0 = auto); `bssid` empty or MAC |
| `clearWifiConfig()` | `CLEAR_WIFI` | — |
| `setMqttConfig({ host, port, username, password, topicPrefix })` | `SET_MQTT_CONFIG` | `host` required; `port` 1-65535; credentials optional |
| `clearMqttConfig()` | `CLEAR_MQTT_CONFIG` | — |
| `setDeviceLabel(label)` | `SET_DEVICE_CONFIG` | single-line string, may be empty |
| `clearDeviceConfig()` | `CLEAR_DEVICE_CONFIG` | — |
| `requestSysinfo()` | `REQ_SYSINFO` | — |
| `writeControl(command)` | any | Escape hatch for commands the library does not model |

The library validates protocol correctness only; product policies (for
example, this site requires a Wi-Fi password even though the protocol allows
open networks) belong to the caller.

### Errors

Command builders throw `ESPectreValidationError` (`error.name ===
'ESPectreValidationError'`); everything else that rejects is a transport or
browser error. `ESPectreBleClient.VERSION` identifies the library version,
independent of the device `proto_version` reported in sysinfo.

### Tests

The hardware-independent parts (command builders, validation, telemetry
parser, event API) are covered by unit tests:

```bash
node --test 'test/web/*.mjs'
```
