# ESPectre - The Game

**A reaction game powered by ESPectre WiFi motion detection technology.**

> Stay still. Move fast. React to survive.

[![Powered by ESPectre](https://img.shields.io/badge/Powered%20by-ESPectre-40DCA5)](https://espectre.dev)
[![License](https://img.shields.io/badge/License-GPLv3-blue)](../../LICENSE)

---

## What is This?

**ESPectre - The Game** is a browser-based reaction game that demonstrates the capabilities of [ESPectre](https://espectre.dev) - a WiFi-based motion detection system.

Instead of using a controller, keyboard, or camera, **your physical movement is detected through WiFi signal interference** analyzed by an ESP32 running ESPectre firmware.

### The Concept

You are a **Spectrum Guardian** - an entity that protects WiFi frequencies from malicious Spectres trying to corrupt them. When an enemy Spectre appears, you must physically move faster than it to dissolve it.

- **Stand still** → You're charging, ready to react
- **Move suddenly** → You attack the Spectre
- **Move too early** → You're detected as a cheater and lose
- **Move too slow** → The enemy hits you first
- **Move harder** → Deal more damage, trigger special effects!

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Browser (https://espectre.dev/game)           ESP32 (BLE)      │
│                                                                 │
│   ┌───────────────────────┐          ┌───────────────────────┐  │
│   │   Game (JavaScript)   │◄────────►│   ESP32 + ESPectre    │  │
│   │                       │   BLE    │                       │  │
│   │   • Web Bluetooth API │          │   • Detects movement  │  │
│   │   • Notify telemetry  │          │   • Sends telemetry   │  │
│   │   • Write controls    │          │   • Sends sysinfo     │  │
│   └───────────────────────┘          └───────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

1. Visit `https://espectre.dev/game` in Chrome or Edge
2. Connect your device via BLE
3. Click "Connect" and grant permission
4. Your physical movement controls the game!

**No backend server required.** The browser communicates directly with the ESP32 via BLE.

---

## Connection Modes

### BLE Mode

Works with ESP32 variants that support BLE:

| Chip | Supported |
|------|-----------|
| ESP32 (classic) | ✅ |
| ESP32-S2 | ❌ |
| ESP32-S3 | ✅ |
| ESP32-C3 | ✅ |
| ESP32-C5 | ✅ |
| ESP32-C6 | ✅ |
| ESP32-H2 | ❌ |

The game is designed for desktop browsers with Web Bluetooth support.

| Aspect | Details |
|--------|---------|
| API | Web Bluetooth |
| Conflict with esphome logs | No |
| Latency | ~10-50ms (depends on notify rate) |

### Mouse Mode (Demo)

For testing without hardware or in unsupported browsers.

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | Vanilla JavaScript + CSS |
| Device channel | Web Bluetooth API (Chrome/Edge) |
| Hosting | GitHub Pages (espectre.dev/game) |
| Backend | None (fully client-side) |

### Browser Support

| Browser | Web Bluetooth | Mouse Mode |
|---------|---------------|------------|
| Chrome 89+ | ✅ | ✅ |
| Edge 89+ | ✅ | ✅ |
| Opera 76+ | ✅ | ✅ |
| Firefox | ❌ | ✅ |
| Safari | ❌ | ✅ |

---

## Communication Protocol

The game uses a BLE protocol with telemetry notifications and control writes.

### UUIDs (Reference Profile)

| Item | UUID | Direction | Notes |
|------|------|-----------|-------|
| Service | `d33ff46b-2203-4775-bc6f-b3a2c36af8f0` | - | ESPectre BLE service |
| Telemetry characteristic | `119d5cac-48da-4bd9-bfc3-169805868258` | ESP32 -> Browser (`notify`) | Binary payload |
| Sysinfo characteristic | `c8c89ffa-c401-461f-9ffc-942fa04adfe3` | ESP32 -> Browser (`notify`) | Text `key=value` lines |
| Control characteristic | `33ed9214-a8d7-40e8-82d1-c82747dcdc71` | Browser -> ESP32 (`write`) | ASCII commands |

### System Info (ESP32 → Browser)

Sent over BLE notify when requested and at client connect.

```
proto_version=1
chip=esp32c6
threshold=1.20 (auto)
window=75
END
```

| Key | Description |
|-----|-------------|
| `chip` | ESP32 chip model (e.g., `esp32c6`) |
| `threshold` | Current motion detection threshold |
| `window` | Segmentation window size (packets) |
| `detector` | Active detector (`MVS` or `ML`) |
| `subcarriers` | Subcarrier selection mode (`yaml` or `auto`) |
| `lowpass` | Low-pass filter status (`on`/`off`) |
| `lowpass_cutoff` | Low-pass cutoff frequency (Hz) |
| `hampel` | Hampel filter status (`on`/`off`) |
| `hampel_window` | Hampel window size |
| `hampel_threshold` | Hampel threshold |
| `traffic_rate` | Traffic generator rate (packets/sec) |
| `publish_interval` | ESPectre publish interval (packets) |
| `best_pxx` | Calibration baseline metric used for adaptive thresholding |
| `proto_version` | Game BLE protocol version |
| `END` | Marks end of system info block |

### Data (ESP32 → Browser)

Sent via BLE `telemetry` characteristic notifications.

```
[float32 movement][float32 threshold]
```

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `movement` | float | Current movement intensity (moving variance, same as Home Assistant sensor) |
| `threshold` | float | Motion detection threshold (from ESPectre config) |

Telemetry uses little-endian `float32` values.

### Control Commands (Browser/Client -> ESP32)

| Command | Description | Limits |
|---------|-------------|--------|
| `REQ_SYSINFO` | Requests a fresh sysinfo block | Exact command string |
| `SET_THRESHOLD:X.XX` | Updates runtime threshold | `X` must be finite and in range `0.0-10.0` |

Notes:
- Threshold updates are runtime/session-only and are recalculated at boot.
- Unknown or invalid commands are ignored by firmware (warning logged on device).
- The BLE protocol is reusable by any standard BLE client, not only this game.

### Frame Examples

Telemetry notification (`movement=0.75`, `threshold=1.20`):

```text
00 00 40 3F  9A 99 99 3F
```

Sysinfo notification sequence:

```text
proto_version=1
chip=esp32c6
threshold=1.20 (auto)
window=75
END
```

Control write examples:

```text
REQ_SYSINFO
SET_THRESHOLD:1.80
```

### Movement Detection

The game uses the same threshold as Home Assistant for motion detection:

- **Cheat detection**: `movement > threshold × 1.0` (moving during WAIT phase)
- **Valid hit**: `movement > threshold × 1.2` (moving during MOVE phase)

### Power Calculation

Hit power determines damage and visual effects:

```javascript
const power = movement / (threshold * moveMultiplier);  // moveMultiplier = 1.2
```

| Power | Hit Strength | Damage |
|-------|--------------|--------|
| < 0.5 | None | 0 |
| 0.5 - 1.0 | Weak | 1 |
| 1.0 - 2.0 | Normal | 1 |
| 2.0 - 3.0 | Strong | 2 |
| 3.0+ | Critical | 3 |

This allows gameplay mechanics like:
- Multi-hit enemies requiring several weak hits
- One-shot kills with powerful movements
- Visual feedback based on hit intensity
- Bonus points for stronger attacks

---

## Gameplay

### Game Flow

```
PHASE 1: WAIT
┌─────────────────────────────────────────┐
│                                         │
│        👻 Enemy Spectre appears         │
│           (materializing...)            │
│                                         │
│         "Stay still..."                 │
│                                         │
│   Movement: ████████░░ Stable           │
│   (Movement now = CHEATER!)             │
│                                         │
└─────────────────────────────────────────┘
              │
              ▼ (2-5 seconds random delay)

PHASE 2: TRIGGER
┌─────────────────────────────────────────┐
│                                         │
│                 "MOVE!"                 │
│                                         │
│        👻💥 Enemy attacks!              │
│                                         │
│       MOVE NOW to counter!              │
│       Timer: ███░░░░░ 450ms             │
│                                         │
└─────────────────────────────────────────┘
                       │
                ┌──────┴──────┐
                ▼             ▼

PHASE 3A: WIN                PHASE 3B: LOSE
┌───────────────────┐       ┌───────────────────┐
│                   │       │                   │
│    DISSOLVED!     │       │    CORRUPTED      │
│                   │       │                   │
│  Time: 287ms      │       │  "Too slow..."    │
│  Power: 2.3x      │       │                   │
│  STRONG HIT!      │       │  [TRY AGAIN]      │
│                   │       │                   │
│  Streak: x5       │       │                   │
└───────────────────┘       └───────────────────┘
```

### Enemy Types (Progression)

| Wave | Spectre | Max Reaction Time | HP | Points |
|------|---------|-------------------|-----|--------|
| 1-3 | **Wisp** | 800ms | 1 | 100 |
| 4-6 | **Shade** | 600ms | 2 | 200 |
| 7-9 | **Phantom** | 450ms | 2 | 350 |
| 10-12 | **Glitch** | 350ms | 3 | 500 |
| 13+ | **Void** | 250ms | 3 | 750 |

Enemies with HP > 1 require multiple hits or one powerful hit (power >= HP).

---

## Mouse Fallback

For testing without an ESP32, move your mouse to simulate motion detection.
Move faster for stronger hits - the velocity of your mouse maps to movement intensity.

---

## System Info Panel

After connecting via BLE, the game displays a **System Info** panel showing the current ESPectre configuration:

| Field | Description |
|-------|-------------|
| Threshold | Motion detection threshold |
| Window | Segmentation window size |
| Subcarriers | Selection mode (YAML config or NBVI auto-calibration) |
| Low-pass | Filter status and cutoff frequency |
| Hampel | Filter status |
| Traffic | Traffic generator rate |

This provides immediate visibility into how the device is configured without needing to check ESPHome logs.

---

## Threshold Tuning

The game doubles as a fun way to tune your ESPectre system. The movement bar at the bottom of the screen shows real-time motion data and the current threshold.

**Drag the threshold marker** to adjust sensitivity:

- Drag **left** → lower ESPectre threshold on device (more sensitive)
- Drag **right** → higher ESPectre threshold on device (less sensitive)

Threshold drag sends a BLE control command (`SET_THRESHOLD:X.XX`) and updates ESPectre runtime threshold for the active session.

### Runtime Controls via BLE

- `SET_THRESHOLD:X.XX` updates detection threshold at runtime (session-only)
- `REQ_SYSINFO` requests a fresh sysinfo block

This provides immediate visual feedback:
- See exactly how your movements register
- Test different positions in the room
- Find the sweet spot between false positives and missed detections
- Verify the system works before relying on it for automation

---

## Related Documentation

| Document | Description |
|----------|-------------|
| [Web Bluetooth API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Bluetooth_API) | Browser Web Bluetooth API (MDN) |

---

## License

This game is part of the ESPectre project and is released under the **GNU General Public License v3.0 (GPLv3)**.

See [LICENSE](../../LICENSE) for the full license text.
