# Changelog - ESPectre Arduino

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0] - 2024-01-XX

### Added
- Initial release of Arduino port
- MVS (Moving Variance Segmentation) detector
- NBVI (Non-consecutive Band Variance Index) calibrator
- Gain lock support for ESP32-S3/C3/C5/C6
- TFT display integration (Adafruit ST7789, 240x135)
- NeoPixel LED status indicator
- Auto-calibration with adaptive threshold (P95 × 1.4)
- Traffic generator (UDP DNS queries at 100 pps)
- Serial debug output
- Hardware test sketch (`HARDWARE_TEST.ino`)
- Comprehensive documentation (README, QUICKSTART)
- PlatformIO support (`platformio.ini`)
- Arduino Library Manager compatibility (`library.properties`)

### Algorithm Implementation
- Spatial turbulence calculation (std dev across subcarriers)
- Moving variance calculation (temporal variance)
- NBVI band selection (12 non-consecutive subcarriers)
- P95-based adaptive threshold

### Hardware Support
- Adafruit Feather ESP32-S3 Reverse TFT (primary target)
- ESP32-S3 with gain lock support
- Compatible with ESP32-C3/C5/C6 (with minor pin changes)
- ESP32 original (no gain lock, reduced accuracy)

### Dependencies
- Adafruit ST7789 v1.10+
- Adafruit GFX Library v1.11+
- Adafruit NeoPixel v1.12+
- Arduino-ESP32 v2.0.14+

### Performance
- ~97% detection accuracy (optimal conditions)
- <1% false positive rate
- 1-2s detection latency
- ~15s boot time (WiFi + gain + calibration)
- 100 pps CSI packet rate

### Known Limitations
- Requires line-of-sight or single wall (multiple walls reduce accuracy)
- Optimal range: 3-8 meters from WiFi router
- 2.4 GHz WiFi only (ESP32 limitation)
- No Home Assistant integration (standalone only)
- No ML detector (MVS only)

## [Unreleased]

### Planned Features
- [ ] ML detector port (neural network inference)
- [ ] MQTT support for Home Assistant integration
- [ ] Web interface for configuration
- [ ] SD card logging
- [ ] Battery optimization (deep sleep)
- [ ] Multiple AP monitoring
- [ ] Gesture recognition
- [ ] Activity classification
- [ ] People counting

### Possible Improvements
- [ ] Configurable display themes
- [ ] Touch button calibration trigger
- [ ] OTA updates via Arduino OTA
- [ ] WiFi provisioning (captive portal)
- [ ] Real-time graphing on TFT
- [ ] Historical data visualization
- [ ] Bluetooth configuration

---

## Version History

### Version Numbering
- **Major.Minor.Patch** (e.g., 1.2.3)
- **Major**: Breaking changes to API or configuration
- **Minor**: New features, backwards compatible
- **Patch**: Bug fixes, documentation updates

### Release Cadence
- Releases as needed based on features/fixes
- Tagged in Git: `arduino-v1.0.0`
- Documented in this CHANGELOG

---

## Contributing

See the main ESPectre repository for contribution guidelines:
- [CONTRIBUTING.md](../CONTRIBUTING.md)
- [GitHub Issues](https://github.com/paulhey/espectre/issues)

When reporting issues specific to the Arduino port, please use the "arduino" label.
