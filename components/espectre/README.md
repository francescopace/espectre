# ESPectre ESPHome Component

Wi-Fi CSI-based motion detection component for ESPHome.

## Features

- ✅ Motion detection using Wi-Fi Channel State Information (CSI)
- ✅ Automatic subcarrier calibration (NBVI algorithm)
- ✅ 10 optional mathematical features
- ✅ Configurable digital filters (Butterworth, Wavelet, Hampel, Savitzky-Golay)
- ✅ Native ESPHome integration (OTA, YAML config, HA integration)
- ✅ Traffic generator for continuous CSI packets

## Installation

Add to your ESPHome YAML configuration:

```yaml
external_components:
  - source: github://francescopace/espectre@porting/esphome
    components: [ espectre ]
    refresh: 5min
```

## Minimal Configuration

```yaml
esphome:
  name: espectre-sensor

esp32:
  variant: ESP32S3  # or ESP32C6
  framework:
    type: esp-idf
    version: 5.4.2
    sdkconfig_options:
      CONFIG_ESP_WIFI_DYNAMIC_RX_BUFFER_NUM: '128'
      CONFIG_ESP_WIFI_DYNAMIC_TX_BUFFER_NUM: '128'
      CONFIG_ESP_WIFI_CSI_ENABLED: y
      CONFIG_ESP_WIFI_AMPDU_TX_ENABLED: n
      CONFIG_ESP_WIFI_AMPDU_RX_ENABLED: n
      CONFIG_ESP_WIFI_STA_DISCONNECTED_PM_ENABLE: n
      CONFIG_PM_ENABLE: n

wifi:
  ssid: !secret wifi_ssid
  password: !secret wifi_password
  power_save_mode: NONE

espectre:
  traffic_generator_rate: 100

binary_sensor:
  - platform: espectre
    name: "Motion Detected"
    device_class: motion

sensor:
  - platform: espectre
    movement:
      name: "Movement Score"
```

## Full Configuration

```yaml
espectre:
  # Traffic generator (ESSENTIAL for CSI!)
  traffic_generator_rate: 100  # packets/sec (0-1000)
  
  # Motion detection
  segmentation_threshold: 1.0  # 0.5-10.0
  segmentation_window_size: 50  # 10-200 packets
  
  # NBVI calibration
  nbvi_auto_calibrate: true
  
  # Features
  features_enabled: true
  
  # Filters
  butterworth_enabled: true
  wavelet_enabled: false
  wavelet_level: 3  # 1-3
  wavelet_threshold: 1.0  # 0.5-2.0
  hampel_enabled: true
  hampel_threshold: 2.0  # 1.0-10.0
  savgol_enabled: true

binary_sensor:
  - platform: espectre
    name: "Motion Detected"
    device_class: motion

sensor:
  - platform: espectre
    # Core sensors
    movement:
      name: "Movement Score"
    threshold:
      name: "Detection Threshold"
    turbulence:
      name: "Signal Turbulence"
    
    # Feature sensors (optional)
    variance:
      name: "CSI Variance"
    skewness:
      name: "CSI Skewness"
    kurtosis:
      name: "CSI Kurtosis"
    entropy:
      name: "CSI Entropy"
    iqr:
      name: "CSI IQR"
    spatial_variance:
      name: "Spatial Variance"
    spatial_correlation:
      name: "Spatial Correlation"
    spatial_gradient:
      name: "Spatial Gradient"
    temporal_delta_mean:
      name: "Temporal Delta Mean"
    temporal_delta_variance:
      name: "Temporal Delta Variance"
```

## Supported Platforms

- ESP32-S3 (WiFi 4)
- ESP32-C6 (WiFi 6 on 2.4 GHz)

## Important Notes

### Traffic Generator

⚠️ **CRITICAL**: The traffic generator is ESSENTIAL for CSI packet generation. Without it, the ESP32 receives zero CSI packets and detection will not work.

- Default: 100 pps (optimal for activity recognition)
- Range: 0-1000 pps
- Set to 0 only if you have other continuous WiFi traffic

### NBVI Auto-Calibration

On first boot (or when no saved configuration exists), the component will automatically:
1. Collect 500 CSI packets (~5 seconds)
2. Analyze baseline characteristics
3. Select optimal 12 subcarriers
4. Save configuration to preferences

**Requirements during calibration:**
- Room must be quiet (no movement)
- Wait 5-10 seconds for completion

### SDK Configuration

The following sdkconfig options are REQUIRED in your YAML:

```yaml
esp32:
  framework:
    type: esp-idf
    sdkconfig_options:
      CONFIG_ESP_WIFI_CSI_ENABLED: y
      CONFIG_ESP_WIFI_AMPDU_TX_ENABLED: n
      CONFIG_ESP_WIFI_AMPDU_RX_ENABLED: n
      CONFIG_PM_ENABLE: n

wifi:
  power_save_mode: NONE
```

## Troubleshooting

### No CSI packets received

1. Verify traffic generator is enabled (`traffic_generator_rate > 0`)
2. Check WiFi is connected
3. Verify SDK configuration options are set
4. Check logs for CSI initialization errors

### Motion detection not working

1. Adjust `segmentation_threshold` (try 0.5-2.0 for more sensitivity)
2. Check sensor placement (optimal: 3-8m from router)
3. Verify traffic generator is active
4. Monitor logs for turbulence values

### Calibration fails

1. Ensure room is quiet during calibration
2. Check traffic generator is running
3. Verify WiFi connection is stable
4. Increase timeout if needed

## License

GPLv3 - See main ESPectre repository for details

## Links

- Main Repository: https://github.com/francescopace/espectre
- Issue #17: https://github.com/francescopace/espectre/issues/17
- Documentation: https://github.com/francescopace/espectre/blob/main/README.md
