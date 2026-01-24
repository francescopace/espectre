"""
ESPectre Component

ESPHome component for ESPectre WiFi CSI-based motion detection.
Sensors are defined directly in the component (not as separate platforms).

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

from pathlib import Path

import esphome.codegen as cg
import esphome.config_validation as cv
from esphome.components import sensor, binary_sensor, number, switch
from esphome.components.esp32 import add_extra_build_file, add_idf_sdkconfig_option
from esphome.const import (
    CONF_ID,
    STATE_CLASS_MEASUREMENT,
    DEVICE_CLASS_MOTION,
    UNIT_EMPTY,
    ENTITY_CATEGORY_CONFIG,
    ICON_PULSE,
)

DEPENDENCIES = ["wifi"]
AUTO_LOAD = ["sensor", "binary_sensor", "number", "switch"]

# Configuration parameters
CONF_SEGMENTATION_THRESHOLD = "segmentation_threshold"
CONF_SEGMENTATION_WINDOW_SIZE = "segmentation_window_size"
CONF_TRAFFIC_GENERATOR_RATE = "traffic_generator_rate"
CONF_PUBLISH_INTERVAL = "publish_interval"
CONF_SELECTED_SUBCARRIERS = "selected_subcarriers"

# Low-pass filter
CONF_LOWPASS_ENABLED = "lowpass_enabled"
CONF_LOWPASS_CUTOFF = "lowpass_cutoff"

# Hampel filter
CONF_HAMPEL_ENABLED = "hampel_enabled"
CONF_HAMPEL_WINDOW = "hampel_window"
CONF_HAMPEL_THRESHOLD = "hampel_threshold"


# Traffic generator mode
CONF_TRAFFIC_GENERATOR_MODE = "traffic_generator_mode"

# Gain lock mode
CONF_GAIN_LOCK = "gain_lock"

# Segmentation calibration (only applies to MVS, ignored by PCA)
CONF_SEGMENTATION_CALIBRATION = "segmentation_calibration"

# Detection algorithm
CONF_DETECTION_ALGORITHM = "detection_algorithm"

# Threshold limits (keep in sync with csi_processor.h)
THRESHOLD_MIN = 0.1
THRESHOLD_MAX = 10.0
THRESHOLD_DEFAULT = 1.0

# Sensors - defined directly in component
CONF_MOVEMENT_SENSOR = "movement_sensor"
CONF_MOTION_SENSOR = "motion_sensor"

# Number controls
CONF_THRESHOLD_NUMBER = "threshold_number"

# Switch controls
CONF_CALIBRATE_SWITCH = "calibrate_switch"

espectre_ns = cg.esphome_ns.namespace("espectre")
ESpectreComponent = espectre_ns.class_("ESpectreComponent", cg.Component)
ESpectreThresholdNumber = espectre_ns.class_("ESpectreThresholdNumber", number.Number, cg.Component)
ESpectreCalibrateSwitch = espectre_ns.class_("ESpectreCalibrateSwitch", switch.Switch, cg.Component)

def validate_segmentation_threshold(value):
    """Validate segmentation_threshold: accepts 'auto', 'min', or a float."""
    if isinstance(value, str):
        value_lower = value.lower()
        if value_lower in ("auto", "min"):
            return value_lower
        # Try to parse as float
        try:
            return float(value)
        except ValueError:
            raise cv.Invalid(f"Invalid threshold value '{value}'. Use 'auto', 'min', or a number {THRESHOLD_MIN}-{THRESHOLD_MAX}")
    if isinstance(value, (int, float)):
        if value < THRESHOLD_MIN or value > THRESHOLD_MAX:
            raise cv.Invalid(f"Threshold must be between {THRESHOLD_MIN} and {THRESHOLD_MAX}")
        return float(value)
    raise cv.Invalid(f"Invalid threshold type. Use 'auto', 'min', or a number {THRESHOLD_MIN}-{THRESHOLD_MAX}")


CONFIG_SCHEMA = cv.Schema({
    cv.GenerateID(): cv.declare_id(ESpectreComponent),
    
    # Motion detection parameters
    # segmentation_threshold:
    #   - auto (default): P95 × 1.4 - low false positives
    #   - min: P100 × 1.0 - maximum sensitivity (may have FP)
    #   - number (0.1-10.0): fixed manual threshold
    cv.Optional(CONF_SEGMENTATION_THRESHOLD, default="auto"): validate_segmentation_threshold,
    cv.Optional(CONF_SEGMENTATION_WINDOW_SIZE, default=50): cv.int_range(min=10, max=200),
    
    # Traffic generator (0 = disabled, use external WiFi traffic)
    cv.Optional(CONF_TRAFFIC_GENERATOR_RATE, default=100): cv.int_range(min=0, max=1000),
    
    # Traffic generator mode: dns (default) or ping (ICMP, more compatible)
    cv.Optional(CONF_TRAFFIC_GENERATOR_MODE, default="dns"): cv.one_of("dns", "ping", lower=True),
    
    # Gain lock mode: auto (default), enabled, or disabled
    # Auto: enables gain lock but skips if signal too strong (AGC < 30)
    # Enabled: always force gain lock (may freeze if too close to AP)
    # Disabled: never lock gain (less stable CSI but works at any distance)
    cv.Optional(CONF_GAIN_LOCK, default="auto"): cv.one_of("auto", "enabled", "disabled", lower=True),
    
    # Segmentation calibration: nbvi (default) or p95 - only applies to MVS detector
    # NBVI: selects 12 non-consecutive subcarriers based on stability metrics
    # P95: selects 12 consecutive subcarriers minimizing P95 moving variance
    # Note: ignored when detection_algorithm is PCA (uses its own calibrator)
    cv.Optional(CONF_SEGMENTATION_CALIBRATION, default="nbvi"): cv.one_of("p95", "nbvi", lower=True),
    
    # Detection algorithm: mvs (Moving Variance) or pca (PCA + correlation)
    cv.Optional(CONF_DETECTION_ALGORITHM, default="mvs"): cv.one_of("mvs", "pca", lower=True),
    
    # Publish interval in packets (default: same as traffic_generator_rate, or 100 if traffic is 0)
    cv.Optional(CONF_PUBLISH_INTERVAL): cv.int_range(min=1, max=1000),
    
    # Subcarrier selection (optional - if not specified, auto-calibrates at every boot)
    cv.Optional(CONF_SELECTED_SUBCARRIERS): cv.All(
        cv.ensure_list(cv.int_range(min=0, max=63)),
        cv.Length(min=1, max=12)
    ),
    
    # Low-pass filter for noise reduction (disabled by default)
    cv.Optional(CONF_LOWPASS_ENABLED, default=False): cv.boolean,
    cv.Optional(CONF_LOWPASS_CUTOFF, default=11.0): cv.float_range(min=5.0, max=20.0),
    
    # Hampel filter for turbulence outlier removal
    cv.Optional(CONF_HAMPEL_ENABLED, default=False): cv.boolean,
    cv.Optional(CONF_HAMPEL_WINDOW, default=7): cv.int_range(min=3, max=11),
    cv.Optional(CONF_HAMPEL_THRESHOLD, default=4.0): cv.float_range(min=1.0, max=10.0),
    
    # Sensors - optional with defaults, always created
    cv.Optional(CONF_MOVEMENT_SENSOR, default={"name": "Movement Score"}): sensor.sensor_schema(
        unit_of_measurement=UNIT_EMPTY,
        accuracy_decimals=2,
        state_class=STATE_CLASS_MEASUREMENT,
    ),
    cv.Optional(CONF_MOTION_SENSOR, default={"name": "Motion Detected"}): binary_sensor.binary_sensor_schema(
        device_class=DEVICE_CLASS_MOTION,
    ),
    
    # Number control for threshold adjustment from HA
    cv.Optional(CONF_THRESHOLD_NUMBER, default={"name": "Threshold"}): number.number_schema(
        ESpectreThresholdNumber,
        entity_category=ENTITY_CATEGORY_CONFIG,
        icon=ICON_PULSE,
    ),
    
    # Switch control for manual recalibration from HA
    # ON = calibrating, OFF = idle. Switch auto-turns off when calibration completes.
    cv.Optional(CONF_CALIBRATE_SWITCH, default={"name": "Calibrate"}): switch.switch_schema(
        ESpectreCalibrateSwitch,
        entity_category=ENTITY_CATEGORY_CONFIG,
    ),
}).extend(cv.COMPONENT_SCHEMA)


def _compute_publish_interval(config):
    """Compute publish_interval default based on traffic_generator_rate."""
    traffic_rate = config[CONF_TRAFFIC_GENERATOR_RATE]
    if CONF_PUBLISH_INTERVAL not in config or config[CONF_PUBLISH_INTERVAL] is None:
        # Default: use traffic rate, or 100 if traffic is disabled
        config[CONF_PUBLISH_INTERVAL] = traffic_rate if traffic_rate > 0 else 100
    return config


FINAL_VALIDATE_SCHEMA = _compute_publish_interval


async def to_code(config):
    var = cg.new_Pvariable(config[CONF_ID])
    await cg.register_component(var, config)
    
    # Add custom partitions.csv with SPIFFS for calibration buffer
    # This allows the component to work without requiring users to manually copy partitions.csv
    partitions_path = Path(__file__).parent / "partitions.csv"
    if partitions_path.exists():
        add_extra_build_file("partitions.csv", partitions_path)
        # Tell PlatformIO to use our custom partition table
        cg.add_platformio_option("board_build.partitions", "partitions.csv")
    
    # Set required sdkconfig options for CSI functionality
    # These are automatically applied - user doesn't need to specify them in YAML
    add_idf_sdkconfig_option("CONFIG_ESP_WIFI_CSI_ENABLED", True)
    add_idf_sdkconfig_option("CONFIG_PM_ENABLE", False)
    add_idf_sdkconfig_option("CONFIG_ESP_WIFI_STA_DISCONNECTED_PM_ENABLE", False)
    
    # CSI optimization options (based on Espressif esp-csi recommendations)
    add_idf_sdkconfig_option("CONFIG_ESP_WIFI_AMPDU_TX_ENABLED", False)
    add_idf_sdkconfig_option("CONFIG_ESP_WIFI_AMPDU_RX_ENABLED", False)
    add_idf_sdkconfig_option("CONFIG_ESP_WIFI_DYNAMIC_RX_BUFFER_NUM", 128)
    # Note: CONFIG_FREERTOS_HZ=1000 is already set by ESPHome
    
    # Configure parameters
    # segmentation_threshold can be: "auto", "min", or a float
    threshold_value = config[CONF_SEGMENTATION_THRESHOLD]
    if isinstance(threshold_value, str):
        # "auto" or "min" mode
        cg.add(var.set_threshold_mode(threshold_value))
    else:
        # Numeric value - set as manual threshold
        cg.add(var.set_segmentation_threshold(threshold_value))
    
    cg.add(var.set_segmentation_window_size(config[CONF_SEGMENTATION_WINDOW_SIZE]))
    cg.add(var.set_traffic_generator_rate(config[CONF_TRAFFIC_GENERATOR_RATE]))
    cg.add(var.set_traffic_generator_mode(config[CONF_TRAFFIC_GENERATOR_MODE]))
    cg.add(var.set_gain_lock_mode(config[CONF_GAIN_LOCK]))
    cg.add(var.set_segmentation_calibration(config[CONF_SEGMENTATION_CALIBRATION]))
    cg.add(var.set_detection_algorithm(config[CONF_DETECTION_ALGORITHM]))
    cg.add(var.set_publish_interval(config[CONF_PUBLISH_INTERVAL]))
    
    # Configure subcarriers if specified
    if CONF_SELECTED_SUBCARRIERS in config:
        cg.add(var.set_selected_subcarriers(config[CONF_SELECTED_SUBCARRIERS]))
    
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
    
    
    sens = await binary_sensor.new_binary_sensor(config[CONF_MOTION_SENSOR])
    cg.add(var.set_motion_binary_sensor(sens))
    
    # Register threshold number control
    # Note: number.new_number() handles component registration internally
    # Do NOT call register_component separately - it causes double initialization
    # that leads to "Load access fault" crash on boot (null pointer in early setup)
    num = await number.new_number(
        config[CONF_THRESHOLD_NUMBER],
        min_value=THRESHOLD_MIN,
        max_value=THRESHOLD_MAX,
        step=0.1,
    )
    cg.add(num.set_parent(var))
    cg.add(var.set_threshold_number(num))
    
    # Register calibrate switch control
    # Note: switch.new_switch() handles component registration internally
    # Do NOT call register_component separately - same reason as above
    sw = await switch.new_switch(config[CONF_CALIBRATE_SWITCH])
    cg.add(sw.set_parent(var))
    cg.add(var.set_calibrate_switch(sw))
