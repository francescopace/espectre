# ESPectre discovery

ESPectre devices announce themselves on the local network with mDNS and DNS-SD over IPv4. Use **Find devices** in the browser tools or `./espectre devices` to list them. If discovery fails, enter the device IP instead.

This reference describes the discovery contract for client and firmware developers. For command options, see the [`devices` command](CLI.md#devices); for the device API, see the [API reference](API.md).

## DNS-SD and mDNS

Every frontend publishes the service `_espectre._tcp.local.` for its Direct HTTP endpoint on TCP port `62587`. The host name is always `espectre-{device_id}.local`. The service name can show the user's label, but clients identify a device by `device_id`.

Clients browse the PTR record, then resolve SRV, TXT, and address records. Only IPv4 (A) addresses are used; a device that resolves only to IPv6 is skipped.

| Frontend | Service type | Direct SRV port | Other frontend service |
| --- | --- | --- | --- |
| Native | `_espectre._tcp.local.` | `62587` | Optional MQTT |
| ESPHome | `_espectre._tcp.local.` | `62587` | ESPHome native API |
| Matter | `_espectre._tcp.local.` | `62587` | Matter operational and commissioning services |
| Micro | `_espectre._tcp.local.` | `62587` | none |

A manually entered address can use another port; clients never try old ports. ESPHome and Matter also publish their own services, but `./espectre devices` looks only for `_espectre._tcp.local.`.

### TXT record

| Key | Value |
| --- | --- |
| `txtvers` | `1` |
| `protovers` | `1.0` |
| `device_id` | 16 lowercase hexadecimal characters |
| `name` | Effective display name |
| `frontend` | `native`, `esphome`, `matter`, or `micro` |
| `transport` | `http` |
| `path` | `/espectre/v1` |
| `firmware` | Running firmware version |
| `chip` | Active target, such as `esp32c3` |
| `capabilities` | Bounded comma-separated discovery hints |

Rules for clients:

- Build all URLs (`/events`, `/csi`, and resources) from `path`, then read `GET /capabilities` for what the device supports. The `capabilities` TXT value is only a display hint, not a permission.
- `txtvers` is the version of this TXT layout. `protovers` is the same protocol version as `capabilities.protocol_version`. An unknown value of either means the device is incompatible. Unknown keys are ignored.
- Compare TXT keys and DNS names without regard to ASCII case. If a key appears twice, the first one wins, even if it is empty. Values keep their case.
- A record is valid only with an IPv4 address, a non-zero port, a valid `device_id`, a known `frontend`, the exact `txtvers`, `protovers`, and `transport` values above, and `path=/espectre/v1`. `name`, `firmware`, `chip`, and `capabilities` are extra information, not identity.

Devices send lowercase keys, with `txtvers` first.

### Service lifecycle

The service is published only while the device has an IPv4 address. It is withdrawn on a clean disconnect and announced again after reconnecting or changing address. On ESPHome and Matter, the platform runs the mDNS responder and ESPectre only adds or removes its own service.

- **Native** updates the TXT `name` when the label changes.
- **ESPHome** uses the same `espectre-{device_id}.local` host name without changing its YAML name or entity IDs.
- **Matter** publishes the service only after commissioning. Removing the last fabric removes it and stops Direct HTTP.

## Browser bootstrap

A web page cannot browse DNS-SD. Instead, the portal resolves a special host name that ESPectre devices answer, then asks one of them to browse for the others. Each attempt uses a new random name (96 bits from Web Crypto, as 24 lowercase hexadecimal characters):

```text
espectre-devices-{nonce}.local
```

A new name every time means no cached answer can be reused. The fixed name `espectre-devices.local` is not supported.

### Bootstrap DNS behavior

Native, ESPHome, and commissioned Matter devices answer the bootstrap name. Micro does not answer it, but still shows up in results found by another device.

- The answer is the device's IPv4 address with a 10-second TTL. Several devices may answer the same name.
- The answer includes an NSEC record saying there is no IPv6 address (see [Limits](#limits)).
- The responder handles compressed names, multiple questions, ANY, QU, multicast, and legacy-unicast queries. It skips records the client already knows, waits for Known Answer continuations, and limits repeated multicast replies. Malformed queries are ignored.
- It keeps at most four pending answers and sends at most eight packets per second. Under load, queries can be dropped; the client then needs another attempt. Pending answers are dropped on disconnect or address change.

### `/devices` scan

Once one device answers, the portal calls `GET /espectre/v1/devices` on it (Native, ESPHome, and Matter). The request takes no parameters and follows the same origin rules as any Direct request.

On the device:

- It browses `_espectre._tcp.local.` for 3,000 ms and returns what it found.
- If a scan is already running, it returns `409` with code `conflict`; if it cannot start one, code `unavailable`.
- It closes the connection after each response, so repeated searches do not use up sockets. It keeps no list of peers afterwards.

In the browser:

- The whole search has a 10-second deadline.
- If the first request has no answer after 4 seconds, or fails at the network level, the browser tries once more with a new random name. The first valid result wins and the other request is cancelled. A `409` from one request does not cancel the other.
- An invalid response, a denied permission, or any other HTTP error ends the search.

The result schema is:

```json
{
  "schema_version": 2,
  "elapsed_ms": 3019,
  "status": "complete",
  "truncated": false,
  "rejected_results": 0,
  "devices": [
    {
      "device_id": "0123456789abcdef",
      "instance": "ESPectre 0123456789abcdef",
      "hostname": "espectre-0123456789abcdef",
      "name": "ESPectre C3 abcdef",
      "frontend": "native",
      "dns_sd_schema_version": 1,
      "protocol_version": "1.0",
      "transport": "http",
      "path": "/espectre/v1",
      "firmware": "3.0.0-rc1",
      "chip": "esp32c3",
      "port": 62587,
      "capabilities": ["config", "csi", "monitor"],
      "addresses": ["192.168.1.29"]
    }
  ]
}
```

The top-level fields have these constraints:

| Field | Type and constraint |
| --- | --- |
| `schema_version` | integer equal to `2` |
| `elapsed_ms` | integer from `0` through `10000`; reports the device-side scan duration |
| `status` | `complete` or `timeout`; a timeout may still carry accepted records |
| `truncated` | boolean; true when a device, address, or serialization limit removed output |
| `rejected_results` | non-negative integer counting invalid records and conflicting identities |
| `devices` | array containing at most eight validated device objects |

Each device object uses this schema:

| Field | Type and constraint |
| --- | --- |
| `device_id` | 16-character lowercase hexadecimal string |
| `instance` | printable ASCII string, 1 to 63 characters |
| `hostname` | 1 to 63 letters, digits, `-`, or `_`, without the `.local` suffix |
| `name` | printable ASCII string, 0 to 63 characters |
| `frontend` | `native`, `esphome`, `matter`, or `micro` |
| `dns_sd_schema_version` | integer equal to `1` |
| `protocol_version` | string equal to `1.0` |
| `transport` | string equal to `http` |
| `path` | string equal to `/espectre/v1` |
| `firmware` | printable ASCII string, 1 to 48 characters |
| `chip` | 1 to 16 letters, digits, `-`, or `_` |
| `port` | integer from `1` through `65535` |
| `capabilities` | 1 to 8 unique tokens, each at most 32 characters |
| `addresses` | 1 to 2 validated on-link IPv4 address strings |

How the device builds the list:

- The answering device always includes itself.
- Devices are merged by `device_id` and sorted by it. Addresses of the same device are merged and sorted numerically. Host names are compared without regard to case, keeping the first spelling.
- If two records for one `device_id` disagree on host name, frontend, port, or path, that device is rejected.
- Only unicast IPv4 addresses on the device's own subnet are kept.

The response contains no credentials, settings, motion data, CSI, or broker details.

### Serialization limits

- The TXT `capabilities` value is at most 128 characters. Tokens use letters, digits, `-`, and `_`; a duplicate token makes the record invalid.
- The whole result is at most 3,584 bytes. When a limit is hit, the first entries in sort order are kept and `truncated` is true.

## Client validation and fallback

The portal checks the whole result before showing any device. It remembers only the address you pick, never the bootstrap name or the list. After you pick a device, it reads `GET /device` and `GET /capabilities` and checks that `device_id`, frontend, protocol version, and base path match the discovery data.

Routers, multicast filtering, client isolation, and browser permissions can block discovery while the device itself is still reachable. In that case, connect with the device IP, its `espectre-{device_id}.local` name, a saved address, or Improv Serial.

## Limits

- Multicast filtering, client isolation, and packet loss can hide a device that still answers on its IP. Devices sometimes went missing on a mesh network. Retries help but cannot guarantee a result; see [mesh Wi-Fi instability](TROUBLESHOOTING.md#mesh-wi-fi-instability).
- Only IPv4 is supported.
- The bootstrap name has several responders but still carries an NSEC record, which deviates from [RFC 6762, section 6.1](https://www.rfc-editor.org/rfc/rfc6762.html#section-6.1). Removing it is still experimental.
- Espressif's own host-name responder has known deviations (invalid RCODE queries, class ANY, negative AAAA answers, and legacy-unicast TTLs). ESPectre's bootstrap responder does not change them.
