import esphome.codegen as cg
import esphome.config_validation as cv
from esphome.components import sensor, binary_sensor
from esphome.const import (
    CONF_ID,
)

DEPENDENCIES = ["wifi"]
AUTO_LOAD = ["sensor", "binary_sensor"]

# Configuration parameters
CONF_SEGMENTATION_THRESHOLD = "segmentation_threshold"
CONF_SEGMENTATION_WINDOW_SIZE = "segmentation_window_size"
CONF_FEATURES_ENABLED = "features_enabled"
CONF_TRAFFIC_GENERATOR_RATE = "traffic_generator_rate"
CONF_SELECTED_SUBCARRIERS = "selected_subcarriers"

# Filters
CONF_BUTTERWORTH_ENABLED = "butterworth_enabled"
CONF_WAVELET_ENABLED = "wavelet_enabled"
CONF_WAVELET_LEVEL = "wavelet_level"
CONF_WAVELET_THRESHOLD = "wavelet_threshold"
CONF_HAMPEL_ENABLED = "hampel_enabled"
CONF_HAMPEL_THRESHOLD = "hampel_threshold"
CONF_SAVGOL_ENABLED = "savgol_enabled"

# Sensors
CONF_MOVEMENT = "movement"
CONF_THRESHOLD = "threshold"
CONF_TURBULENCE = "turbulence"
CONF_MOTION = "motion"

# Feature sensors
CONF_VARIANCE = "variance"
CONF_SKEWNESS = "skewness"
CONF_KURTOSIS = "kurtosis"
CONF_ENTROPY = "entropy"
CONF_IQR = "iqr"
CONF_SPATIAL_VARIANCE = "spatial_variance"
CONF_SPATIAL_CORRELATION = "spatial_correlation"
CONF_SPATIAL_GRADIENT = "spatial_gradient"
CONF_TEMPORAL_DELTA_MEAN = "temporal_delta_mean"
CONF_TEMPORAL_DELTA_VARIANCE = "temporal_delta_variance"

espectre_ns = cg.esphome_ns.namespace("espectre")
ESpectreComponent = espectre_ns.class_("ESpectreComponent", cg.Component)

CONFIG_SCHEMA = cv.Schema({
    cv.GenerateID(): cv.declare_id(ESpectreComponent),
    
    # Motion detection
    cv.Optional(CONF_SEGMENTATION_THRESHOLD, default=1.0): cv.float_range(min=0.5, max=10.0),
    cv.Optional(CONF_SEGMENTATION_WINDOW_SIZE, default=50): cv.int_range(min=10, max=200),
    
    # Traffic generator (ESSENTIAL!)
    cv.Optional(CONF_TRAFFIC_GENERATOR_RATE, default=100): cv.int_range(min=0, max=1000),
    
    # Features
    cv.Optional(CONF_FEATURES_ENABLED, default=True): cv.boolean,
    
    # Subcarrier selection (optional - if not specified, auto-calibrates at every boot)
    cv.Optional(CONF_SELECTED_SUBCARRIERS): cv.All(
        cv.ensure_list(cv.int_range(min=0, max=63)),
        cv.Length(min=1, max=12)
    ),
    
    # Filters
    cv.Optional(CONF_BUTTERWORTH_ENABLED, default=True): cv.boolean,
    cv.Optional(CONF_WAVELET_ENABLED, default=False): cv.boolean,
    cv.Optional(CONF_WAVELET_LEVEL, default=3): cv.int_range(min=1, max=3),
    cv.Optional(CONF_WAVELET_THRESHOLD, default=1.0): cv.float_range(min=0.5, max=2.0),
    cv.Optional(CONF_HAMPEL_ENABLED, default=True): cv.boolean,
    cv.Optional(CONF_HAMPEL_THRESHOLD, default=2.0): cv.float_range(min=1.0, max=10.0),
    cv.Optional(CONF_SAVGOL_ENABLED, default=True): cv.boolean,
}).extend(cv.COMPONENT_SCHEMA)

async def to_code(config):
    var = cg.new_Pvariable(config[CONF_ID])
    await cg.register_component(var, config)
    
    # Add C++ source files to build
    cg.add_library(
        "espectre_core",
        None,
        [
            "csi_features.cpp",
            "filters.cpp",
            "wavelet.cpp",
            "csi_processor.cpp",
            "traffic_generator.cpp",
            "nbvi_calibrator.cpp",
        ]
    )
    
    # Configure parameters
    cg.add(var.set_segmentation_threshold(config[CONF_SEGMENTATION_THRESHOLD]))
    cg.add(var.set_segmentation_window_size(config[CONF_SEGMENTATION_WINDOW_SIZE]))
    cg.add(var.set_traffic_generator_rate(config[CONF_TRAFFIC_GENERATOR_RATE]))
    cg.add(var.set_features_enabled(config[CONF_FEATURES_ENABLED]))
    
    # Configure subcarriers if specified
    if CONF_SELECTED_SUBCARRIERS in config:
        cg.add(var.set_selected_subcarriers(config[CONF_SELECTED_SUBCARRIERS]))
    
    # Configure filters
    cg.add(var.set_butterworth_enabled(config[CONF_BUTTERWORTH_ENABLED]))
    cg.add(var.set_wavelet_enabled(config[CONF_WAVELET_ENABLED]))
    cg.add(var.set_wavelet_level(config[CONF_WAVELET_LEVEL]))
    cg.add(var.set_wavelet_threshold(config[CONF_WAVELET_THRESHOLD]))
    cg.add(var.set_hampel_enabled(config[CONF_HAMPEL_ENABLED]))
    cg.add(var.set_hampel_threshold(config[CONF_HAMPEL_THRESHOLD]))
    cg.add(var.set_savgol_enabled(config[CONF_SAVGOL_ENABLED]))
