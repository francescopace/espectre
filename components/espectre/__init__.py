"""
ESPectre Component

ESPHome component for ESPectre WiFi CSI-based motion detection.
Sensors are defined directly in the component (not as separate platforms).

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import esphome.codegen as cg
import esphome.config_validation as cv
from esphome.components import sensor, binary_sensor, number
from esphome.const import (
    CONF_ID,
    STATE_CLASS_MEASUREMENT,
    DEVICE_CLASS_MOTION,
    UNIT_EMPTY,
    ENTITY_CATEGORY_CONFIG,
    ICON_PULSE,
)

DEPENDENCIES = ["wifi"]
AUTO_LOAD = ["sensor", "binary_sensor", "number"]

# Configuration parameters
CONF_SEGMENTATION_THRESHOLD = "segmentation_threshold"
CONF_SEGMENTATION_WINDOW_SIZE = "segmentation_window_size"
CONF_TRAFFIC_GENERATOR_RATE = "traffic_generator_rate"
CONF_SELECTED_SUBCARRIERS = "selected_subcarriers"

# Hampel filter
CONF_HAMPEL_ENABLED = "hampel_enabled"
CONF_HAMPEL_WINDOW = "hampel_window"
CONF_HAMPEL_THRESHOLD = "hampel_threshold"

# Sensors - defined directly in component
CONF_MOVEMENT_SENSOR = "movement_sensor"
CONF_MOTION_SENSOR = "motion_sensor"

# Number controls
CONF_THRESHOLD_NUMBER = "threshold_number"

espectre_ns = cg.esphome_ns.namespace("espectre")
ESpectreComponent = espectre_ns.class_("ESpectreComponent", cg.Component)
ESpectreThresholdNumber = espectre_ns.class_("ESpectreThresholdNumber", number.Number, cg.Component)

CONFIG_SCHEMA = cv.Schema({
    cv.GenerateID(): cv.declare_id(ESpectreComponent),
    
    # Motion detection parameters
    cv.Optional(CONF_SEGMENTATION_THRESHOLD, default=1.0): cv.float_range(min=0.5, max=10.0),
    cv.Optional(CONF_SEGMENTATION_WINDOW_SIZE, default=50): cv.int_range(min=10, max=200),
    
    # Traffic generator (ESSENTIAL!)
    cv.Optional(CONF_TRAFFIC_GENERATOR_RATE, default=100): cv.int_range(min=0, max=1000),
    
    # Subcarrier selection (optional - if not specified, auto-calibrates at every boot)
    cv.Optional(CONF_SELECTED_SUBCARRIERS): cv.All(
        cv.ensure_list(cv.int_range(min=0, max=63)),
        cv.Length(min=1, max=12)
    ),
    
    # Hampel filter for turbulence outlier removal
    cv.Optional(CONF_HAMPEL_ENABLED, default=True): cv.boolean,
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
}).extend(cv.COMPONENT_SCHEMA)

async def to_code(config):
    var = cg.new_Pvariable(config[CONF_ID])
    await cg.register_component(var, config)
    
    # Configure parameters
    cg.add(var.set_segmentation_threshold(config[CONF_SEGMENTATION_THRESHOLD]))
    cg.add(var.set_segmentation_window_size(config[CONF_SEGMENTATION_WINDOW_SIZE]))
    cg.add(var.set_traffic_generator_rate(config[CONF_TRAFFIC_GENERATOR_RATE]))
    
    # Configure subcarriers if specified
    if CONF_SELECTED_SUBCARRIERS in config:
        cg.add(var.set_selected_subcarriers(config[CONF_SELECTED_SUBCARRIERS]))
    
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
    num = await number.new_number(
        config[CONF_THRESHOLD_NUMBER],
        min_value=0.5,
        max_value=10.0,
        step=0.1,
    )
    cg.add(num.set_parent(var))
    cg.add(var.set_threshold_number(num))
