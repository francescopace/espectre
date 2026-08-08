# ADR: keep HT20 sensing while making the Wi-Fi band explicit

- Status: Accepted
- Date: 2026-08-05
- Updated: 2026-08-07

## Context

The shared CSI radio policy originally forced `WIFI_BAND_MODE_2G_ONLY`, then set the 2.4 GHz `11b/g/n` protocol bitmap and `WIFI_BW_HT20`. The first version of this decision replaced the forced band with `WIFI_BAND_MODE_AUTO` on every dual-band target.

Only the PHY ceiling and channel width are sensing requirements. The current detectors consume one 64-subcarrier HT20 channel estimate: `HT20_NUM_SUBCARRIERS` is 64 in `csi_format.h`, the shared live band is the 56 populated bins of that grid, and `DEFAULT_SUBCARRIERS` indexes it directly. An HT20 PPDU has the same layout on 2.4 GHz and 5 GHz.

Band selection is a deployment policy. It changes propagation, AP selection, and the physical conditions seen by the detector, and only ESP32-C5 among the published targets supports 5 GHz. Choosing AUTO globally would therefore change the operating conditions on C5 without integrator intent, even though the production corpus currently validates 2.4 GHz only.

VHT20 and HE20 must not be grouped together. VHT20 still uses a 64-point FFT over a 20 MHz channel and has the same 56 active-tone positions as HT20, so it may be a relatively small extension. It is nevertheless a distinct PHY and LTF capture path that has not been validated by ESPectre. HE20 instead uses a 256-point FFT with 242 active tones and requires a new mapping and evidence.

## Decision

Keep **HT20 as the only production sensing contract on every selected band**, and expose band selection to the embedding frontend or integrator.

- `2g` is the default on every target. It preserves the band covered by the production detector corpus.
- `5g` is an explicit ESP32-C5-only choice.
- `auto` is an explicit ESP32-C5-only choice when the deployment intentionally allows either band.
- Single-band targets reject `5g` and `auto` rather than silently falling back.
- A configured channel hint must belong to the selected band.

The policy is represented by `WifiBandPolicy` in the published `RuntimeConfig` and by a shared ESP-IDF Kconfig choice. Native, Matter, and Streamer take the Kconfig value; SDK integrators set `RuntimeConfig::wifi_band_policy` directly. ESPHome remains the owner of Wi-Fi association and exposes the choice through its native `wifi.band_mode` property. The ESPectre component derives its runtime policy from that validated value instead of exposing a second, potentially conflicting YAML property. When it is omitted on ESP32-C5, the component follows ESPHome's native AUTO default; single-band targets remain fixed to 2.4 GHz. The shipped ESPHome C5 examples select 2.4 GHz explicitly because that is the characterized band.

On the selected band, the lifecycle pins the newest permitted PHY to 802.11n and the bandwidth to HT20:

- 2.4 GHz uses `11b/g/n`.
- 5 GHz uses `11a/n`, excluding 802.11ac and 802.11ax for now.
- AUTO configures both of those per-band protocol sets and HT20 bandwidth on both bands.

On dual-band silicon, the band mode is applied before protocol and bandwidth. Fixed-band policies use the single-band ESP-IDF APIs after selecting that band; AUTO uses the per-band APIs, which are required under `WIFI_BAND_MODE_AUTO`. Failure to apply the explicit band policy is fatal to the CSI lifecycle, because associating on a different band would violate the integrator's requested operating conditions.

The optional station channel remains an association hint, not a guarantee that the AP will stay on that channel. BSSID plus channel is the repeatable-capture configuration; runtime channel changes still invalidate the sensing session.

## Validation

The packet-level gate remains unchanged: `csi_rx_is_ht20_sensing()` checks PHY format and 20 MHz width, and `assess_ht20_sensing_format()` rejects everything outside the current HT20 contract. Unsupported PHY and width packets are counted rather than silently sensed.

Host coverage compiles `wifi_lifecycle.cpp` for both single-band and forced dual-band capabilities. It verifies:

- the default policy selects 2.4 GHz, `11b/g/n`, and HT20;
- an explicit 5 GHz policy selects 5 GHz, `11a/n`, and HT20;
- AUTO configures both bands through the per-band APIs;
- band selection happens before the protocol ceiling;
- an unsupported policy or failed band-mode write fails closed;
- channel hints are accepted only when compatible with the selected band; and
- protocol or bandwidth drift is corrected without redundant writes when the requested state is already active.

Real ESP32-C5 firmware and paired 2.4/5 GHz detection quality still require hardware validation. Until that evidence exists, 5 GHz is an explicit, uncharacterized option rather than the default.

## Alternatives Considered

### Force 2.4 GHz without an integrator option

Rejected. It preserves the validated default but prevents a legitimate HT20 deployment on the ESP32-C5's 5 GHz radio.

### Select AUTO on every dual-band target

Rejected. AUTO delegates a material detector operating condition to AP and station selection, can move an ESP32-C5 onto the uncharacterized 5 GHz path, and overwrites an integrator's deliberate band choice.

### Leave the band mode untouched

Rejected. Behavior would depend on the ESP-IDF or application default and could silently vary between frontends. An explicit policy is testable and makes ownership clear.

### Accept VHT20 now

Deferred to v3.x validation. Its 20 MHz subcarrier grid suggests reuse may be possible, but production support still requires enabling and classifying the VHT-LTF capture path, preserving correct PHY provenance, proving the mapping, and validating both detectors with a representative corpus.

### Accept HE20 now

Deferred to later v3.x research. Its 242-tone, 256-point layout is not the current 64-bin detector contract. Supporting it requires an explicit mapping or a separate sensing contract, parity work, datasets, and detector validation.

## Consequences

- Existing deployments retain their 2.4 GHz behavior by default.
- ESP32-C5 integrations can deliberately select `5g` or `auto` without changing the production HT20 detector contract.
- The integrator owns band selection; the shared runtime owns the HT20 PHY ceiling and fail-closed packet validation.
- 802.11ac/VHT and 802.11ax/HE remain excluded from production capture until their separate v3.x gates are satisfied.
- A 5 GHz link may use a DFS channel. Radar-driven AP channel changes surface as a sensing-session reset rather than corrupted evidence.
- Runtime diagnostics report protocol, bandwidth, channel, and associated band.

## Follow-up

- Collect a paired 5 GHz HT20 corpus on ESP32-C5 and compare it against the 2.4 GHz baseline before making detection-quality claims.
- Validate VHT20 as the first possible post-HT20 production extension in v3.x.
- Study HE20 later in v3.x as a distinct mapping and detector-contract problem.
- Keep HT40 and wider layouts as separate research paths.

See [`ROADMAP.md`](../ROADMAP.md) for the staged research gates.

## Related

- [`2026-07-19-preserve-per-record-phy-provenance-in-streamer-datasets.md`](2026-07-19-preserve-per-record-phy-provenance-in-streamer-datasets.md)
- [`2026-06-09-replace-runtime-nbvi-with-fixed-shared-subcarriers.md`](2026-06-09-replace-runtime-nbvi-with-fixed-shared-subcarriers.md)
- [`ALGORITHMS.md`](../ALGORITHMS.md)
