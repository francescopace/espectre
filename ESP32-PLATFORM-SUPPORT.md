# ESP32 Platform Support Summary

## Supported Platforms

ESPectre now supports multiple ESP32 platforms with optimized configurations for each:

### ESP32-S3 ✅
- **Status**: Fully tested and supported
- **CPU**: Dual-core Xtensa LX7 @ 240MHz
- **Memory**: 8MB PSRAM
- **WiFi**: 802.11 b/g/n (2.4 GHz only)
- **CSI Config**: `lltf_en`, `htltf_en` (legacy structure)
- **CSI Data**: 256 bytes (L-LTF + HT-LTF)
- **Subcarriers**: 64 total (order: 0~31, -32~-1)
- **USB**: Stable, no issues

### ESP32-C6 ✅
- **Status**: Fully tested and supported
- **CPU**: Single-core RISC-V @ 160MHz
- **Memory**: No PSRAM
- **WiFi**: 802.11 b/g/n/ax (WiFi 6 tested on 2.4 GHz)
- **CSI Config**: `acquire_csi_legacy`, `acquire_csi_ht20`, `acquire_csi_ht40`, `acquire_csi_su`
- **CSI Data**: 128-256 bytes (HT20: 128, HT40: 256)
- **Subcarriers**: 64 (HT20) or 128 (HT40), effective range varies
- **Advantages**: WiFi 6, HT40 support, higher CSI packet rate, more subcarriers

### ESP32 (Classic) ❌
- **Status**: Not tested, not officially supported
- **Note**: May work but requires testing and validation

## Key Differences

| Feature | ESP32-S3 | ESP32-C6 |
|---------|----------|----------|
| **Performance** | Better (dual-core) | Good (single-core) |
| **Memory** | 8MB PSRAM | No PSRAM |
| **CSI Packet Rate** | 50-100 pps | 50-100+ pps (WiFi 6) |
| **CSI Configuration** | Simple (`lltf_en`) | Complex (`acquire_csi_*`) |
| **Subcarrier Count** | 64 | 64 (HT20) or 128 (HT40) |
| **WiFi Standard** | WiFi 4 (802.11n) | WiFi 6 (802.11ax) |

## Configuration Files

Each platform has its own optimized configuration:

- `sdkconfig.defaults.esp32s3`: ESP32-S3 configuration
- `sdkconfig.defaults.esp32c6`: ESP32-C6 configuration

## Code Adaptations

### Platform-Specific Code

The codebase uses conditional compilation for platform-specific features:

```c
#if CONFIG_IDF_TARGET_ESP32C6
    // ESP32-C6 specific code
#else
    // ESP32-S3 specific code
#endif
```

### Files with Platform-Specific Code

1. **main/espectre.c**: CSI configuration (different structures)
2. **main/csi_processor.c**: Subcarrier filtering ranges

### Platform-Agnostic Modules

These modules work identically on all platforms:
- `segmentation.c/h`
- `filters.c/h`
- `wavelet.c/h`
- `mqtt_handler.c/h`
- `traffic_generator.c/h`
- `config_manager.c/h`
- `nvs_storage.c/h`

## Migration Guide

### From ESP32-S3 to ESP32-C6

1. Copy correct sdkconfig: `cp sdkconfig.defaults.esp32c6 sdkconfig.defaults`
2. Clean build: `idf.py fullclean`
3. Build: `idf.py build`
4. Flash (may require multiple attempts): `idf.py flash`
5. Monitor: `idf.py monitor`
6. Verify CSI packets are received (should see "🔔 CSI CALLBACK" logs)

### Known Issues

**ESP32-C6:**
- USB flashing instability (use lower baud rates or flash script)

## Performance Comparison

Based on testing:

| Metric | ESP32-S3 | ESP32-C6 |
|--------|----------|----------|
| CSI Packet Rate | 50-100 pps | 50-100+ pps |
| Motion Detection | Excellent | Excellent |
| CPU Usage | Low (dual-core) | Moderate (single-core) |
| Memory Usage | Low (with PSRAM) | Higher (no PSRAM) |

## Recommendations

### Choose ESP32-S3 if:
- ✅ You need more computing power to run small models directly on the device
- ✅ You need PSRAM for future features

### Choose ESP32-C6 if:
- ✅ You want WiFi 6 support
- ✅ You want higher CSI packet rates

## Future Work

### ESP32-C6 Optimization
- [ ] Compare detection accuracy with ESP32-S3 baseline

### Documentation
- [ ] Performance benchmarks

## ESP32-C6 CSI Configuration Details

### Problem Solved

ESP32-C6 CSI callback was never invoked due to incomplete configuration. The issue was resolved by properly configuring the `wifi_csi_acquire_config_t` structure with all required fields.

### Root Cause

ESP32-C6 uses a different CSI configuration structure than ESP32-S3. Simply setting `.enable = 1` is insufficient - you must explicitly specify which CSI types to acquire using `acquire_csi_*` fields.

### Working Configuration

```c
#if CONFIG_IDF_TARGET_ESP32C6
    wifi_csi_config_t csi_config = {
        .enable = 1,                    // Master enable (REQUIRED)
        .acquire_csi_legacy = 1,        // CRITICAL: Required for callback!
        .acquire_csi_ht20 = 1,          // CRITICAL: Required for HT20 packets!
        .acquire_csi_ht40 = 0,          // Disabled: Captures HT40 packets (128 subcarriers)
        .acquire_csi_su = 1,            // Enabled: WiFi 6 Single-User support
        .acquire_csi_mu = 0,            // Disabled: WiFi 6 Multi-User
        .acquire_csi_dcm = 0,           // Disabled: DCM
        .acquire_csi_beamformed = 0,    // Disabled: Beamformed
        .acquire_csi_he_stbc = 0,       // Disabled: HE STBC
        .val_scale_cfg = 0,             // Auto-scaling
        .dump_ack_en = 0,               // Disabled: ACK frames
    };
#endif
```

### Data Validation

According to ESP-IDF issue [#14271](https://github.com/espressif/esp-idf/issues/14271), validate CSI data using the `rx_channel_estimate_info_vld` field in `wifi_pkt_rx_ctrl_t` to filter out invalid CSI data.

## References

- ESP-IDF WiFi API Guide ([ESP32-C6](https://docs.espressif.com/projects/esp-idf/en/latest/esp32c6/api-guides/wifi.html), [ESP32-C6](https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/api-guides/wifi.html))