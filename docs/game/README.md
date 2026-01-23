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
│   Browser (https://espectre.dev/game)         ESP32 (USB)       │
│                                                                 │
│   ┌───────────────────────┐          ┌───────────────────────┐  │
│   │   Game (JavaScript)   │◄────────►│   ESP32 + ESPectre    │  │
│   │                       │   USB    │                       │  │
│   │   • Web Serial API    │          │   • Detects movement  │  │
│   │   • ~100 Hz updates   │          │   • Sends motion data │  │
│   │                       │          │   • On-demand mode    │  │
│   └───────────────────────┘          └───────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

1. Visit `https://espectre.dev/game` in Chrome or Edge
2. Connect your device via USB cable
3. Click "Connect" and grant permission
4. Your physical movement controls the game!

**No backend server required.** The browser communicates directly with the ESP32 via USB.

---

## Connection Modes

### USB Serial Mode

Works with ESP32 variants that have native USB Serial JTAG support:

| Chip | Supported |
|------|-----------|
| ESP32 (classic) | ❌ No native USB |
| ESP32-S2 | ✅ |
| ESP32-S3 | ✅ |
| ESP32-C3 | ✅ |
| ESP32-C5 | ✅ |
| ESP32-C6 | ✅ |
| ESP32-H2 | ✅ |

The classic ESP32 lacks native USB and uses UART via an external USB-to-serial chip, which cannot receive commands from the browser's Web Serial API while ESPHome logs are active.

| Aspect | Details |
|--------|---------|
| API | Web Serial |
| Conflict with esphome logs | Yes (close logs first) |
| Latency | ~1-5ms |

### Mouse Mode (Demo)

For testing without hardware or in unsupported browsers.

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | Vanilla JavaScript + CSS |
| USB | Web Serial API (Chrome/Edge) |
| Hosting | GitHub Pages (espectre.dev/game) |
| Backend | None (fully client-side) |

### Browser Support

| Browser | Web Serial | Mouse Mode |
|---------|------------|------------|
| Chrome 89+ | ✅ | ✅ |
| Edge 89+ | ✅ | ✅ |
| Opera 76+ | ✅ | ✅ |
| Firefox | ❌ | ✅ |
| Safari | ❌ | ✅ |

---

## Communication Protocol

The game uses an on-demand protocol - data is only sent when the browser requests it.

### Commands (Browser → ESP32)

| Command | Description |
|---------|-------------|
| `START\n` | Start streaming game data (also triggers system info) |
| `STOP\n` | Stop streaming game data |
| `PING\n` | Keep-alive signal (sent every 2 seconds) |
| `T:X.XX\n` | Set threshold (e.g., `T:1.50\n`) |

The ESP32 automatically stops streaming if no `PING` is received for 5 seconds. This handles browser crashes or sudden disconnections.

### System Info (ESP32 → Browser)

Sent once when `START` command is received. Used to display device configuration in the UI.

```
[I][espectre:NNN][espectre]: [sysinfo] key=value
```

| Key | Description |
|-----|-------------|
| `chip` | ESP32 chip model (e.g., `esp32c6`) |
| `threshold` | Current motion detection threshold |
| `window` | Segmentation window size (packets) |
| `subcarriers` | Subcarrier selection mode (`yaml` or `auto`) |
| `lowpass` | Low-pass filter status (`on`/`off`) |
| `lowpass_cutoff` | Low-pass cutoff frequency (Hz) |
| `hampel` | Hampel filter status (`on`/`off`) |
| `traffic_rate` | Traffic generator rate (packets/sec) |
| `adaptive_thr` | Adaptive threshold value |
| `END` | Marks end of system info |

### Data (ESP32 → Browser)

Sent at ~100 Hz (every CSI packet) while streaming is active. Uses ESPHome log format for clean line separation.

```
[I][stream:NNN][espectre]: <movement>,<threshold>
```

Example: `[I][stream:075][espectre]: 0.73,1.50`

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `movement` | float | Current movement intensity (moving variance, same as Home Assistant sensor) |
| `threshold` | float | Motion detection threshold (from ESPectre config) |

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

After connecting via USB, the game displays a **System Info** panel showing the current ESPectre configuration:

| Field | Description |
|-------|-------------|
| Threshold | Motion detection threshold |
| Window | Segmentation window size |
| Subcarriers | Selection mode (YAML config or P95 auto-calibration) |
| Low-pass | Filter status and cutoff frequency |
| Hampel | Filter status |
| Traffic | Traffic generator rate |

This provides immediate visibility into how the device is configured without needing to check ESPHome logs.

---

## Threshold Tuning

The game doubles as a fun way to tune your ESPectre system. The movement bar at the bottom of the screen shows real-time motion data and the current threshold.

**Drag the threshold marker** to adjust sensitivity:

- Drag **left** → Lower threshold = more sensitive (detects smaller movements)
- Drag **right** → Higher threshold = less sensitive (requires larger movements)

Changes are sent to the ESP32 in real-time and apply to Home Assistant as well. Note: adjustments are session-only - the adaptive threshold is recalculated on every boot.

This provides immediate visual feedback:
- See exactly how your movements register
- Test different positions in the room
- Find the sweet spot between false positives and missed detections
- Verify the system works before relying on it for automation

---

## Related Documentation

| Document | Description |
|----------|-------------|
| [Web Serial API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Serial_API) | Browser Web Serial API (MDN) |

---

## License

This game is part of the ESPectre project and is released under the **GNU General Public License v3.0 (GPLv3)**.

See [LICENSE](../../LICENSE) for the full license text.
