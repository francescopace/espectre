"""
ESPectre - ESPHome Component

ESPHome component for ESPectre WiFi CSI-based motion detection.
Sensors are defined directly in the component (not as separate platforms).

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

from pathlib import Path
import re

import esphome.codegen as cg
import esphome.config_validation as cv
from esphome.components import sensor, binary_sensor, button, number, select, switch
from esphome.components.esp32 import (
    add_idf_sdkconfig_option,
    const as esp32_const,
    get_esp32_variant,
)
from esphome.components.wifi import CONF_BAND_MODE
from esphome.const import (
    CONF_ID,
    CONF_WIFI,
    STATE_CLASS_MEASUREMENT,
    DEVICE_CLASS_MOTION,
    UNIT_EMPTY,
    UNIT_PERCENT,
    UNIT_DECIBEL_MILLIWATT,
    ENTITY_CATEGORY_CONFIG,
    ENTITY_CATEGORY_DIAGNOSTIC,
    ICON_PULSE,
    DEVICE_CLASS_SIGNAL_STRENGTH,
)
from esphome.core import CORE

DEPENDENCIES = ["wifi"]
AUTO_LOAD = ["sensor", "binary_sensor", "button", "number", "select", "switch"]

# Configuration parameters
CONF_SEGMENTATION_WINDOW_SIZE = "segmentation_window_size"
CONF_TRAFFIC_GENERATOR_RATE = "traffic_generator_rate"
CONF_TRAFFIC_GENERATOR_ADAPTIVE = "traffic_generator_adaptive"
CONF_PUBLISH_INTERVAL = "publish_interval"
CONF_EVALUATION_INTERVAL = "evaluation_interval"
CONF_MOTION_ON_HITS = "motion_on_hits"
CONF_MOTION_OFF_HITS = "motion_off_hits"

# Low-pass filter
CONF_LOWPASS_ENABLED = "lowpass_enabled"
CONF_LOWPASS_CUTOFF = "lowpass_cutoff"

# Hampel filter
CONF_HAMPEL_ENABLED = "hampel_enabled"
CONF_HAMPEL_WINDOW = "hampel_window"
CONF_HAMPEL_THRESHOLD = "hampel_threshold"


# Traffic generator mode
CONF_TRAFFIC_GENERATOR_MODE = "traffic_generator_mode"

# Detection algorithm
CONF_DETECTION_ALGORITHM = "detection_algorithm"
CONF_DEBUG_TELEMETRY = "debug_telemetry"

# Sensors - defined directly in component
CONF_MOVEMENT_SENSOR = "movement_sensor"
CONF_INTENSITY_SENSOR = "intensity_sensor"
CONF_MOTION_SENSOR = "motion_sensor"
CONF_TRAFFIC_RATE_SENSOR = "traffic_rate_sensor"
CONF_CSI_CALLBACK_RATE_SENSOR = "csi_callback_rate_sensor"
CONF_CSI_ACCEPTED_RATE_SENSOR = "csi_accepted_rate_sensor"
CONF_CSI_FILTERED_RATE_SENSOR = "csi_filtered_rate_sensor"
CONF_WIFI_CHANNEL_SENSOR = "wifi_channel_sensor"
CONF_WIFI_RSSI_SENSOR = "wifi_rssi_sensor"
CONF_DIAGNOSTICS_BUTTON = "diagnostics_button"

# Number controls
CONF_THRESHOLD_NUMBER = "threshold_number"
CONF_DETECTOR_SELECT = "detector_select"

# Switch controls
CONF_CALIBRATE_SWITCH = "calibrate_switch"

espectre_ns = cg.esphome_ns.namespace("espectre_component")
ESpectreComponent = espectre_ns.class_("ESpectreComponent", cg.Component)
ESpectreThresholdNumber = espectre_ns.class_("ESpectreThresholdNumber", number.Number, cg.Component)
ESpectreDetectorSelect = espectre_ns.class_("ESpectreDetectorSelect", select.Select, cg.Component)
ESpectreCalibrateSwitch = espectre_ns.class_("ESpectreCalibrateSwitch", switch.Switch, cg.Component)
ESpectreDiagnosticsButton = espectre_ns.class_("ESpectreDiagnosticsButton", button.Button, cg.Component)

_LIBRARY_ROOT = Path(__file__).resolve().parents[4]
_SCHEMA_HEADER = _LIBRARY_ROOT / "runtime" / "runtime_sensing_schema.h"
_SCHEMA_CONST_PATTERN = re.compile(
    r"constexpr\s+(?:const char \*const|bool|float|uint8_t|uint16_t|uint32_t)\s+"
    r"(RUNTIME_[A-Z0-9_]+)\s*=\s*([^;]+);"
)

_WIFI_BAND_POLICY_BY_MODE = {
    "AUTO": "auto",
    "2.4GHZ": "2g",
    "5GHZ": "5g",
}


def _library_uri(path: Path) -> str:
    """Return a PlatformIO-compatible file URI for local libraries."""
    return path.resolve().as_uri()


def _parse_schema_literal(raw_value):
    raw_value = raw_value.strip()
    if raw_value in ("true", "false"):
        return raw_value == "true"
    if raw_value.startswith('"') and raw_value.endswith('"'):
        return raw_value[1:-1]
    if raw_value.endswith(("f", "F", "u", "U")):
        raw_value = raw_value[:-1]
    if "." in raw_value or "e" in raw_value.lower():
        return float(raw_value)
    return int(raw_value)


def _load_runtime_schema(schema_path: Path):
    constants = {}
    for line in schema_path.read_text(encoding="utf-8").splitlines():
        match = _SCHEMA_CONST_PATTERN.search(line)
        if match is None:
            continue
        constants[match.group(1)] = _parse_schema_literal(match.group(2))
    return constants


_RUNTIME_SCHEMA = _load_runtime_schema(_SCHEMA_HEADER)

THRESHOLD_MIN = _RUNTIME_SCHEMA["RUNTIME_THRESHOLD_MIN"]
THRESHOLD_MAX = _RUNTIME_SCHEMA["RUNTIME_ML_THRESHOLD_MAX"]
SEGMENTATION_WINDOW_SIZE_DEFAULT = _RUNTIME_SCHEMA["RUNTIME_SEGMENTATION_WINDOW_SIZE_DEFAULT"]
SEGMENTATION_WINDOW_SIZE_MIN = _RUNTIME_SCHEMA["RUNTIME_SEGMENTATION_WINDOW_SIZE_MIN"]
SEGMENTATION_WINDOW_SIZE_MAX = _RUNTIME_SCHEMA["RUNTIME_SEGMENTATION_WINDOW_SIZE_MAX"]
TRAFFIC_GENERATOR_RATE_DEFAULT = _RUNTIME_SCHEMA["RUNTIME_TRAFFIC_GENERATOR_RATE_DEFAULT"]
TRAFFIC_GENERATOR_RATE_MIN = _RUNTIME_SCHEMA["RUNTIME_TRAFFIC_GENERATOR_RATE_MIN"]
TRAFFIC_GENERATOR_RATE_MAX = _RUNTIME_SCHEMA["RUNTIME_TRAFFIC_GENERATOR_RATE_MAX"]
TRAFFIC_GENERATOR_ADAPTIVE_DEFAULT = _RUNTIME_SCHEMA["RUNTIME_TRAFFIC_GENERATOR_ADAPTIVE_DEFAULT"]
TRAFFIC_GENERATOR_MODE_DEFAULT = _RUNTIME_SCHEMA["RUNTIME_TRAFFIC_GENERATOR_MODE_DEFAULT_NAME"]
DETECTION_ALGORITHM_DEFAULT = _RUNTIME_SCHEMA["RUNTIME_DETECTION_ALGORITHM_DEFAULT_NAME"]
PUBLISH_INTERVAL_DEFAULT = _RUNTIME_SCHEMA["RUNTIME_PUBLISH_INTERVAL_DEFAULT"]
EVALUATION_INTERVAL_DEFAULT = _RUNTIME_SCHEMA["RUNTIME_EVALUATION_INTERVAL_DEFAULT"]
INTERVAL_MIN = _RUNTIME_SCHEMA["RUNTIME_INTERVAL_MIN"]
INTERVAL_MAX = _RUNTIME_SCHEMA["RUNTIME_INTERVAL_MAX"]
MOTION_HITS_MIN = _RUNTIME_SCHEMA["RUNTIME_MOTION_HITS_MIN"]
MOTION_HITS_MAX = _RUNTIME_SCHEMA["RUNTIME_MOTION_HITS_MAX"]
MOTION_ON_HITS_DEFAULT = _RUNTIME_SCHEMA["RUNTIME_MOTION_ON_HITS_DEFAULT"]
MOTION_OFF_HITS_DEFAULT = _RUNTIME_SCHEMA["RUNTIME_MOTION_OFF_HITS_DEFAULT"]
LOWPASS_ENABLED_DEFAULT = _RUNTIME_SCHEMA["RUNTIME_LOWPASS_ENABLED_DEFAULT"]
LOWPASS_CUTOFF_DEFAULT = _RUNTIME_SCHEMA["RUNTIME_LOWPASS_CUTOFF_DEFAULT"]
LOWPASS_CUTOFF_MIN = _RUNTIME_SCHEMA["RUNTIME_LOWPASS_CUTOFF_MIN"]
LOWPASS_CUTOFF_MAX = _RUNTIME_SCHEMA["RUNTIME_LOWPASS_CUTOFF_MAX"]
HAMPEL_ENABLED_DEFAULT = _RUNTIME_SCHEMA["RUNTIME_HAMPEL_ENABLED_DEFAULT"]
HAMPEL_WINDOW_DEFAULT = _RUNTIME_SCHEMA["RUNTIME_HAMPEL_WINDOW_DEFAULT"]
HAMPEL_WINDOW_MIN = _RUNTIME_SCHEMA["RUNTIME_HAMPEL_WINDOW_MIN"]
HAMPEL_WINDOW_MAX = _RUNTIME_SCHEMA["RUNTIME_HAMPEL_WINDOW_MAX"]
HAMPEL_THRESHOLD_DEFAULT = _RUNTIME_SCHEMA["RUNTIME_HAMPEL_THRESHOLD_DEFAULT"]
HAMPEL_THRESHOLD_MIN = _RUNTIME_SCHEMA["RUNTIME_HAMPEL_THRESHOLD_MIN"]
HAMPEL_THRESHOLD_MAX = _RUNTIME_SCHEMA["RUNTIME_HAMPEL_THRESHOLD_MAX"]


CONFIG_SCHEMA = cv.Schema({
    cv.GenerateID(): cv.declare_id(ESpectreComponent),
    
    # Motion detection parameters
    cv.Optional(CONF_SEGMENTATION_WINDOW_SIZE, default=SEGMENTATION_WINDOW_SIZE_DEFAULT): cv.int_range(
        min=SEGMENTATION_WINDOW_SIZE_MIN, max=SEGMENTATION_WINDOW_SIZE_MAX
    ),
    # Traffic generator (0 = disabled, use external WiFi traffic)
    cv.Optional(CONF_TRAFFIC_GENERATOR_RATE, default=TRAFFIC_GENERATOR_RATE_DEFAULT): cv.int_range(
        min=TRAFFIC_GENERATOR_RATE_MIN, max=TRAFFIC_GENERATOR_RATE_MAX
    ),
    cv.Optional(CONF_TRAFFIC_GENERATOR_ADAPTIVE, default=TRAFFIC_GENERATOR_ADAPTIVE_DEFAULT): cv.boolean,
    
    # Traffic generator mode: ping (default) or dns
    cv.Optional(CONF_TRAFFIC_GENERATOR_MODE, default=TRAFFIC_GENERATOR_MODE_DEFAULT): cv.one_of(
        "dns", "ping", lower=True
    ),
    
    # Detection algorithm: classic (default) or ml
    # CLASSIC: weighted L1 + autocorrelation fusion - adaptive threshold
    # ML: Machine Learning (MLP neural network) - higher accuracy, fixed subcarriers
    cv.Optional(CONF_DETECTION_ALGORITHM, default=DETECTION_ALGORITHM_DEFAULT): cv.one_of("classic", "ml", lower=True),
    # Internal benchmark switch for shared runtime debug telemetry.
    cv.Optional(CONF_DEBUG_TELEMETRY, default=False): cv.boolean,
    cv.Optional(CONF_EVALUATION_INTERVAL, default=EVALUATION_INTERVAL_DEFAULT): cv.int_range(
        min=INTERVAL_MIN, max=INTERVAL_MAX
    ),
    cv.Optional(CONF_MOTION_ON_HITS, default=MOTION_ON_HITS_DEFAULT): cv.int_range(
        min=MOTION_HITS_MIN, max=MOTION_HITS_MAX
    ),
    cv.Optional(CONF_MOTION_OFF_HITS, default=MOTION_OFF_HITS_DEFAULT): cv.int_range(
        min=MOTION_HITS_MIN, max=MOTION_HITS_MAX
    ),

    cv.Optional(CONF_PUBLISH_INTERVAL, default=PUBLISH_INTERVAL_DEFAULT): cv.int_range(
        min=INTERVAL_MIN, max=INTERVAL_MAX
    ),
    
    # Low-pass filter for noise reduction (disabled by default)
    cv.Optional(CONF_LOWPASS_ENABLED, default=LOWPASS_ENABLED_DEFAULT): cv.boolean,
    cv.Optional(CONF_LOWPASS_CUTOFF, default=LOWPASS_CUTOFF_DEFAULT): cv.float_range(
        min=LOWPASS_CUTOFF_MIN, max=LOWPASS_CUTOFF_MAX
    ),
    
    # Hampel filter for turbulence outlier removal
    cv.Optional(CONF_HAMPEL_ENABLED, default=HAMPEL_ENABLED_DEFAULT): cv.boolean,
    cv.Optional(CONF_HAMPEL_WINDOW, default=HAMPEL_WINDOW_DEFAULT): cv.int_range(
        min=HAMPEL_WINDOW_MIN, max=HAMPEL_WINDOW_MAX
    ),
    cv.Optional(CONF_HAMPEL_THRESHOLD, default=HAMPEL_THRESHOLD_DEFAULT): cv.float_range(
        min=HAMPEL_THRESHOLD_MIN, max=HAMPEL_THRESHOLD_MAX
    ),
    
    # Sensors - optional with defaults, always created
    cv.Optional(CONF_MOVEMENT_SENSOR, default={"name": "Movement Score"}): sensor.sensor_schema(
        unit_of_measurement=UNIT_EMPTY,
        accuracy_decimals=2,
        state_class=STATE_CLASS_MEASUREMENT,
    ),
    cv.Optional(CONF_INTENSITY_SENSOR, default={"name": "Intensity"}): sensor.sensor_schema(
        unit_of_measurement=UNIT_PERCENT,
        accuracy_decimals=1,
        state_class=STATE_CLASS_MEASUREMENT,
        icon="mdi:gauge",
    ),
    cv.Optional(CONF_MOTION_SENSOR, default={"name": "Motion Detected"}): binary_sensor.binary_sensor_schema(
        device_class=DEVICE_CLASS_MOTION,
    ),

    # On-demand diagnostic entities. The component refreshes its internal
    # sample from the existing sensing callback, but publishes these states
    # only when the diagnostics button is pressed.
    cv.Optional(CONF_TRAFFIC_RATE_SENSOR, default={"name": "Traffic TX Rate"}): sensor.sensor_schema(
        unit_of_measurement="pkt/s",
        accuracy_decimals=1,
        state_class=STATE_CLASS_MEASUREMENT,
        entity_category=ENTITY_CATEGORY_DIAGNOSTIC,
        icon="mdi:upload-network",
    ),
    cv.Optional(CONF_CSI_CALLBACK_RATE_SENSOR, default={"name": "CSI Callback Rate"}): sensor.sensor_schema(
        unit_of_measurement="pkt/s",
        accuracy_decimals=1,
        state_class=STATE_CLASS_MEASUREMENT,
        entity_category=ENTITY_CATEGORY_DIAGNOSTIC,
        icon="mdi:access-point",
    ),
    cv.Optional(CONF_CSI_ACCEPTED_RATE_SENSOR, default={"name": "CSI Accepted Rate"}): sensor.sensor_schema(
        unit_of_measurement="pkt/s",
        accuracy_decimals=1,
        state_class=STATE_CLASS_MEASUREMENT,
        entity_category=ENTITY_CATEGORY_DIAGNOSTIC,
        icon="mdi:check-network",
    ),
    cv.Optional(CONF_CSI_FILTERED_RATE_SENSOR, default={"name": "CSI Filtered Rate"}): sensor.sensor_schema(
        unit_of_measurement="pkt/s",
        accuracy_decimals=1,
        state_class=STATE_CLASS_MEASUREMENT,
        entity_category=ENTITY_CATEGORY_DIAGNOSTIC,
        icon="mdi:filter-outline",
    ),
    cv.Optional(CONF_WIFI_CHANNEL_SENSOR, default={"name": "WiFi Channel"}): sensor.sensor_schema(
        accuracy_decimals=0,
        entity_category=ENTITY_CATEGORY_DIAGNOSTIC,
        icon="mdi:wifi-marker",
    ),
    cv.Optional(CONF_WIFI_RSSI_SENSOR, default={"name": "WiFi RSSI"}): sensor.sensor_schema(
        unit_of_measurement=UNIT_DECIBEL_MILLIWATT,
        accuracy_decimals=0,
        device_class=DEVICE_CLASS_SIGNAL_STRENGTH,
        state_class=STATE_CLASS_MEASUREMENT,
        entity_category=ENTITY_CATEGORY_DIAGNOSTIC,
    ),
    cv.Optional(CONF_DIAGNOSTICS_BUTTON, default={"name": "Refresh Diagnostics"}): button.button_schema(
        ESpectreDiagnosticsButton,
        entity_category=ENTITY_CATEGORY_DIAGNOSTIC,
        icon="mdi:refresh",
    ),
    
    # Number control for threshold adjustment from HA
    cv.Optional(CONF_THRESHOLD_NUMBER, default={"name": "Threshold"}): number.number_schema(
        ESpectreThresholdNumber,
        entity_category=ENTITY_CATEGORY_CONFIG,
        icon=ICON_PULSE,
    ),

    cv.Optional(CONF_DETECTOR_SELECT, default={"name": "Detector"}): select.select_schema(
        ESpectreDetectorSelect,
        entity_category=ENTITY_CATEGORY_CONFIG,
    ),
    
    # Switch control for manual recalibration from HA
    # ON = calibrating, OFF = idle. Switch auto-turns off when calibration completes.
    cv.Optional(CONF_CALIBRATE_SWITCH, default={"name": "Calibrate"}): switch.switch_schema(
        ESpectreCalibrateSwitch,
        entity_category=ENTITY_CATEGORY_CONFIG,
    ),
}).extend(cv.COMPONENT_SCHEMA)


def _runtime_wifi_band_policy():
    if get_esp32_variant() != esp32_const.VARIANT_ESP32C5:
        return "2g"

    # ESPHome defaults ESP32-C5 to AUTO when wifi.band_mode is omitted.
    band_mode = str(CORE.config[CONF_WIFI].get(CONF_BAND_MODE, "AUTO"))
    return _WIFI_BAND_POLICY_BY_MODE[band_mode]


async def to_code(config):
    cg.add_library("espectre-shared", None, _library_uri(_LIBRARY_ROOT))

    # PlatformIO compiles the shared library without the ESPHome source tree on
    # its include path, so espectre_log.h would fall back to vanilla esp_log,
    # which ESPHome builds strip below ERROR (CONFIG_LOG_DEFAULT_LEVEL=ERROR).
    # Expose the ESPHome headers so shared runtime logs reach the ESPHome logger.
    cg.add_build_flag("-Isrc")

    var = cg.new_Pvariable(config[CONF_ID])
    await cg.register_component(var, config)

    # Set required sdkconfig options for CSI functionality
    # These are automatically applied - user doesn't need to specify them in YAML
    add_idf_sdkconfig_option("CONFIG_ESP_WIFI_CSI_ENABLED", True)
    add_idf_sdkconfig_option("CONFIG_PM_ENABLE", False)
    add_idf_sdkconfig_option("CONFIG_ESP_WIFI_STA_DISCONNECTED_PM_ENABLE", False)
    
    # CSI optimization options (based on Espressif esp-csi recommendations)
    add_idf_sdkconfig_option("CONFIG_ESP_WIFI_AMPDU_TX_ENABLED", False)
    add_idf_sdkconfig_option("CONFIG_ESP_WIFI_AMPDU_RX_ENABLED", False)
    add_idf_sdkconfig_option("CONFIG_ESP_WIFI_DYNAMIC_RX_BUFFER_NUM", 128)
    if config[CONF_DEBUG_TELEMETRY]:
        cg.add_build_flag("-DCONFIG_ESPECTRE_DEBUG_TELEMETRY=1")
    # Note: CONFIG_FREERTOS_HZ=1000 is already set by ESPHome
    
    # Threshold is selected automatically at startup and remains adjustable
    # through the runtime number control.
    cg.add(var.set_segmentation_window_size(config[CONF_SEGMENTATION_WINDOW_SIZE]))
    # ESPHome owns association policy through wifi.band_mode. Mirror that
    # validated choice into the shared runtime so its HT20 radio setup uses the
    # matching fixed-band or per-band ESP-IDF APIs.
    cg.add(var.set_wifi_band_policy(_runtime_wifi_band_policy()))
    cg.add(var.set_traffic_generator_rate(config[CONF_TRAFFIC_GENERATOR_RATE]))
    cg.add(var.set_traffic_generator_adaptive(config[CONF_TRAFFIC_GENERATOR_ADAPTIVE]))
    cg.add(var.set_traffic_generator_mode(config[CONF_TRAFFIC_GENERATOR_MODE]))
    cg.add(var.set_detection_algorithm(config[CONF_DETECTION_ALGORITHM]))
    cg.add(var.set_publish_interval(config[CONF_PUBLISH_INTERVAL]))
    cg.add(var.set_evaluation_interval(config[CONF_EVALUATION_INTERVAL]))
    cg.add(var.set_motion_on_hits(config[CONF_MOTION_ON_HITS]))
    cg.add(var.set_motion_off_hits(config[CONF_MOTION_OFF_HITS]))
    
    # Configure Low-pass filter
    cg.add(var.set_lowpass_enabled(config[CONF_LOWPASS_ENABLED]))
    cg.add(var.set_lowpass_cutoff(config[CONF_LOWPASS_CUTOFF]))
    
    # Configure Hampel filter
    cg.add(var.set_hampel_enabled(config[CONF_HAMPEL_ENABLED]))
    cg.add(var.set_hampel_window(config[CONF_HAMPEL_WINDOW]))
    cg.add(var.set_hampel_threshold(config[CONF_HAMPEL_THRESHOLD]))
    
    # Register sensors (required, always present)
    sens = await sensor.new_sensor(config[CONF_MOVEMENT_SENSOR])
    cg.add(var.set_movement_sensor(sens))

    sens = await sensor.new_sensor(config[CONF_INTENSITY_SENSOR])
    cg.add(var.set_intensity_sensor(sens))

    sens = await binary_sensor.new_binary_sensor(config[CONF_MOTION_SENSOR])
    cg.add(var.set_motion_binary_sensor(sens))

    diagnostic_sensors = (
        (CONF_TRAFFIC_RATE_SENSOR, var.set_traffic_rate_sensor),
        (CONF_CSI_CALLBACK_RATE_SENSOR, var.set_csi_callback_rate_sensor),
        (CONF_CSI_ACCEPTED_RATE_SENSOR, var.set_csi_accepted_rate_sensor),
        (CONF_CSI_FILTERED_RATE_SENSOR, var.set_csi_filtered_rate_sensor),
        (CONF_WIFI_CHANNEL_SENSOR, var.set_wifi_channel_sensor),
        (CONF_WIFI_RSSI_SENSOR, var.set_wifi_rssi_sensor),
    )
    for config_key, setter in diagnostic_sensors:
        sens = await sensor.new_sensor(config[config_key])
        cg.add(setter(sens))

    diagnostics_button = await button.new_button(config[CONF_DIAGNOSTICS_BUTTON])
    cg.add(diagnostics_button.set_parent(var))
    
    # Register threshold number control
    # Note: number.new_number() handles component registration internally
    # Do NOT call register_component separately - it causes double initialization
    # that leads to "Load access fault" crash on boot (null pointer in early setup)
    threshold_step = 0.01
    num = await number.new_number(
        config[CONF_THRESHOLD_NUMBER],
        min_value=THRESHOLD_MIN,
        max_value=THRESHOLD_MAX,
        step=threshold_step,
    )
    cg.add(num.set_parent(var))
    cg.add(var.set_threshold_number(num))

    detector = await select.new_select(
        config[CONF_DETECTOR_SELECT],
        options=["classic", "ml"],
    )
    cg.add(detector.set_parent(var))
    cg.add(var.set_detector_select(detector))
    
    # Register calibrate switch control
    # Note: switch.new_switch() handles component registration internally
    # Do NOT call register_component separately - same reason as above
    sw = await switch.new_switch(config[CONF_CALIBRATE_SWITCH])
    cg.add(sw.set_parent(var))
    cg.add(var.set_calibrate_switch(sw))
