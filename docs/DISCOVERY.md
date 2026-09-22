# ESPectre Discovery

ESPectre implements IPv4 Zeroconf service discovery using mDNS and DNS-SD.

Use **Find devices** in the portal or `./espectre devices` to find ESPectre devices on your local network. Discovery requires IPv4 and working mDNS between the client and devices. If automatic discovery fails, enter the device IP directly. See [CLI.md](CLI.md#devices) for command options and [API.md](API.md) for the device API.

## DNS-SD and mDNS

ESPectre uses mDNS and DNS-SD to locate the Direct HTTP endpoint. Every networked frontend publishes `_espectre._tcp.local.` on TCP port `62587`. The stable host name is `espectre-{device_id}.local`; the service instance and display name may use the configured label, but consumers use `device_id` as identity.

A DNS-SD browse starts from the service-type PTR record. Each instance resolves through SRV, TXT, and address records. The host CLI accepts IPv4 A records; an advertisement that resolves only through AAAA is excluded.

| Frontend | Service type | Direct SRV port | Other frontend service |
| --- | --- | --- | --- |
| Native | `_espectre._tcp.local.` | `62587` | Optional MQTT |
| ESPHome | `_espectre._tcp.local.` | `62587` | ESPHome native API |
| Matter | `_espectre._tcp.local.` | `62587` | Matter operational and commissioning services |
| Micro | `_espectre._tcp.local.` | `62587` | none |

A manually entered Direct endpoint may specify another port. Clients do not probe legacy ports. ESPHome and Matter keep publishing their upstream service records, but `./espectre devices` browses only `_espectre._tcp.local.`.

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

There is no `events` TXT key. Clients derive resource, `/events`, and `/csi` URLs from `path`, then negotiate the exact surface through `GET /capabilities`. Discovery capability tokens are presentation hints, not authorization or UI feature gates.

`txtvers` versions the TXT key/value schema. `protovers` is the same application version exposed as `capabilities.protocol_version`; it is not an independent Direct version. Unknown TXT keys may be ignored, but an unknown `txtvers` or `protovers` value is incompatible.

Consumers compare TXT keys and DNS names case-insensitively for ASCII letters. The first occurrence of a TXT key wins, including an empty or valueless occurrence; unknown keys are ignored. Values and displayed names retain their case. Advertisements keep their canonical lowercase keys. `txtvers` is the first transmitted TXT entry on all four frontends.

The CLI accepts a record only when it has an IPv4 address, a non-zero SRV port, a valid `device_id`, a supported `frontend`, the exact version and transport values above, and `path=/espectre/v1`. `name`, `firmware`, `chip`, and `capabilities` enrich the result but do not identify the device.

### Service lifecycle

Services are available only while the station interface has a usable IPv4 address. A frontend that owns its mDNS responder sends a best-effort goodbye on a clean disconnect and announces again after reconnect or an address change. ESPHome and Matter retain ownership of their responder lifecycle; ESPectre adds or removes only its service.

Native uses `espectre-{device_id}.local` and updates the TXT `name` after a saved label change. ESPHome uses the same stable ESPectre host identity without changing its YAML name, native API identity, or entity IDs. Matter publishes ESPectre only after a fabric has been commissioned; removing the last fabric removes the service and stops Direct HTTP.

## Browser bootstrap

Web pages cannot enumerate DNS-SD services. For each automatic discovery attempt, the portal obtains 96 random bits from Web Crypto, encodes them as 24 lowercase hexadecimal characters, and starts with a fresh host name:

```text
espectre-devices-{nonce}.local
```

A fresh name prevents a cached positive or negative answer from satisfying a later attempt. The static alias `espectre-devices.local` is unsupported, and firmware does not provide a compatibility fallback for a different bootstrap contract.

### Bootstrap DNS behavior

Native, ESPHome, and commissioned Matter devices answer bootstrap queries. Micro publishes its DNS-SD service and can appear in a peer's discovery result, but does not answer bootstrap names itself.

A bootstrap answer contains the responder's current IPv4 address with a ten-second TTL. Several devices may answer the same name. The response also includes NSEC to indicate that no IPv6 address is available; see the remaining limitation below.

The responder accepts compressed names, multiple questions, ANY queries, multicast, QU, and legacy-unicast requests. It suppresses records already known by the requester, waits for Known Answer continuations in truncated queries, and limits repeated multicast replies. Malformed queries are ignored. Pending responses are cleared after a disconnect or address change.

The responder keeps at most four pending responses and sends at most eight datagrams per second. Its bounded queues can drop queries under contention or overload. These limits and network packet loss can require another discovery attempt.

### `/devices` scan

After resolving one bootstrap responder, the portal requests `GET /espectre/v1/devices` with the same Origin policy as any Direct request and a 10-second client timeout. Native, ESPHome, and Matter implement this resource.

If the first request is still pending after four seconds, the browser tries one more fresh nonce. A transport failure starts that retry immediately. Both requests share the original ten-second deadline; the first valid result wins, and the other request is cancelled. An HTTP `409` from an overlapping scan leaves the other request active. Invalid responses, denied permissions, and other HTTP errors end discovery. Requests to an explicit device address stay on that address.

The request takes no parameters. It starts one asynchronous PTR browse for `_espectre._tcp.local.` with a fixed 3,000 ms query window. A concurrent scan returns HTTP `409` with code `conflict`; a scan that cannot start returns code `unavailable`. Closing the requesting connection prevents later delivery and creates no waiter or persistent peer inventory.

The server closes each `/devices` connection after sending the response so repeated searches do not exhaust the device's socket limit. Other API requests, SSE, and CSI keep their existing connection policy.

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

The result includes the responding device even when the underlying Espressif query API omits its own advertisement. Devices are deduplicated by `device_id` and sorted lexicographically. Records for one identity and endpoint merge their addresses; hostname comparison ignores ASCII case while preserving the first spelling. Records that give one identity conflicting hostnames, frontends, ports, or paths reject that identity. Addresses sort numerically.

Accepted addresses must be IPv4 unicast addresses on the responder's station subnet. Unspecified, network, broadcast, loopback, multicast, and off-link addresses are rejected. The response contains no credentials, configuration secrets, motion events, CSI, or broker details.

### Serialization limits

The comma-separated TXT capability value may contain at most 128 characters. Capability tokens contain only letters, digits, `-`, and `_`; duplicate tokens invalidate the record. The complete result object is limited to 3,584 bytes. Device, address, and output-size limits retain the deterministic leading results and set `truncated`.

## Client validation and fallback

The portal validates the complete result before rendering a device or constructing an endpoint. It remembers only the selected unique address, never the shared bootstrap name or peer list. After selection, the client requests `GET /device` and `GET /capabilities`; the `device_id`, frontend, protocol version, and base path must agree with discovery.

If no eligible responder is reachable, connect with a private device IP, the unique `espectre-{device_id}.local` host name, a remembered endpoint, or Improv Serial. Routed networks, multicast filtering, client isolation, and browser local-network permissions can block discovery without blocking Direct connectivity.

## Limits

- Multicast filtering, client isolation, and packet loss can hide a device that remains reachable by IP. Intermittent omissions were observed on a mesh network. Retries improve recovery but cannot guarantee discovery; use the device IP when needed. See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#mesh-wi-fi-instability).
- Discovery uses IPv4. An IPv6-only advertisement is excluded.
- The shared bootstrap name retains an NSEC assertion despite having multiple responders, a known deviation from [RFC 6762, section 6.1](https://www.rfc-editor.org/rfc/rfc6762.html#section-6.1). Removing NSEC has not completed validation across the supported frontends and remains experimental.
- Espressif's canonical-host responder has known deviations involving invalid RCODE queries, class ANY, negative AAAA answers, and legacy-unicast TTLs. These are outside the ESPectre bootstrap corrections.
