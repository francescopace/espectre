# ESPectre - The Game

**A reaction game powered by ESPectre WiFi motion detection technology.**

> Stay still. Move fast. React to survive.

[![Powered by ESPectre](https://img.shields.io/badge/Powered%20by-ESPectre-40DCA5)](https://espectre.dev)
[![License](https://img.shields.io/badge/License-GPLv3-blue)](../../../LICENSE)

---

## What is This?

**ESPectre - The Game** is a browser-based reaction game that demonstrates the capabilities of [ESPectre](https://espectre.dev) - a WiFi-based motion detection system.

Instead of using a controller, keyboard, or camera, **your physical movement is detected through WiFi signal interference** analyzed by an ESP32 running the standalone ESPectre native frontend firmware.

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
| ESP32 (classic) | Yes |
| ESP32-S3 | Yes |
| ESP32-C3 | Yes |
| ESP32-C5 | Yes |
| ESP32-C6 | Yes |
| ESP32-H2 | No |

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
| Chrome 89+ | Yes | Yes |
| Edge 89+ | Yes | Yes |
| Opera 76+ | Yes | Yes |
| Firefox | No | Yes |
| Safari | No | Yes |

---

## Communication Protocol

The game is one example client built on the standalone ESPectre native frontend.

The protocol source of truth lives in [`ESPECTRE_PROTOCOL.md`](../../ESPECTRE_PROTOCOL.md).
Native frontend workflow and firmware-specific notes live in
[`README.md` (native)](../../../src/cpp/frontend/native/README.md).

This game uses the native frontend over BLE in a straightforward way:

- subscribe to telemetry notifications
- subscribe to sysinfo notifications
- request a fresh sysinfo block when needed
- adjust the runtime threshold from the browser UI

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
│        Enemy Spectre appears            │
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
│        Enemy attacks!                   │
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

After connecting via BLE, the game displays a **System Info** panel showing the
current ESPectre configuration and diagnostics exposed by the native frontend.

For the exact field set and current semantics, see
[`ESPECTRE_PROTOCOL.md`](../../ESPECTRE_PROTOCOL.md).

---

## Threshold Tuning

The game doubles as a fun way to tune your ESPectre system. The movement bar at the bottom of the screen shows real-time motion data and the current threshold.

**Drag the threshold marker** to adjust sensitivity:

- Drag **left** → lower ESPectre threshold on device (more sensitive)
- Drag **right** → higher ESPectre threshold on device (less sensitive)

Threshold drag sends the BLE runtime-threshold control defined in
[`ESPECTRE_PROTOCOL.md`](../../ESPECTRE_PROTOCOL.md) and updates the
ESPectre threshold for the active session.

### Runtime Controls via BLE

- the browser can update the runtime threshold for the active session
- the browser can request a fresh sysinfo block when it needs to refresh the
  panel

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

## Firmware Requirement

Use the dedicated `native` frontend firmware from the web flasher or build it locally with `./espectre native ...`.
The ESPHome frontend no longer embeds this custom BLE protocol.

See [LICENSE](../../../LICENSE) for the full license text.
