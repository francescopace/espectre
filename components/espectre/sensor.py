import esphome.codegen as cg
import esphome.config_validation as cv
from esphome.components import sensor
from esphome.const import (
    CONF_ID,
    STATE_CLASS_MEASUREMENT,
    UNIT_EMPTY,
)
from . import (
    espectre_ns,
    ESpectreComponent,
    CONF_MOVEMENT,
    CONF_THRESHOLD,
    CONF_TURBULENCE,
    CONF_VARIANCE,
    CONF_SKEWNESS,
    CONF_KURTOSIS,
    CONF_ENTROPY,
    CONF_IQR,
    CONF_SPATIAL_VARIANCE,
    CONF_SPATIAL_CORRELATION,
    CONF_SPATIAL_GRADIENT,
    CONF_TEMPORAL_DELTA_MEAN,
    CONF_TEMPORAL_DELTA_VARIANCE,
)

DEPENDENCIES = ["espectre"]

CONFIG_SCHEMA = cv.Schema({
    cv.GenerateID(): cv.use_id(ESpectreComponent),
    
    # Core sensors
    cv.Optional(CONF_MOVEMENT): sensor.sensor_schema(
        unit_of_measurement=UNIT_EMPTY,
        accuracy_decimals=2,
        state_class=STATE_CLASS_MEASUREMENT,
    ),
    cv.Optional(CONF_THRESHOLD): sensor.sensor_schema(
        unit_of_measurement=UNIT_EMPTY,
        accuracy_decimals=2,
        state_class=STATE_CLASS_MEASUREMENT,
    ),
    cv.Optional(CONF_TURBULENCE): sensor.sensor_schema(
        unit_of_measurement=UNIT_EMPTY,
        accuracy_decimals=2,
        state_class=STATE_CLASS_MEASUREMENT,
    ),
    
    # Feature sensors (optional)
    cv.Optional(CONF_VARIANCE): sensor.sensor_schema(
        unit_of_measurement=UNIT_EMPTY,
        accuracy_decimals=3,
        state_class=STATE_CLASS_MEASUREMENT,
    ),
    cv.Optional(CONF_SKEWNESS): sensor.sensor_schema(
        unit_of_measurement=UNIT_EMPTY,
        accuracy_decimals=3,
        state_class=STATE_CLASS_MEASUREMENT,
    ),
    cv.Optional(CONF_KURTOSIS): sensor.sensor_schema(
        unit_of_measurement=UNIT_EMPTY,
        accuracy_decimals=3,
        state_class=STATE_CLASS_MEASUREMENT,
    ),
    cv.Optional(CONF_ENTROPY): sensor.sensor_schema(
        unit_of_measurement=UNIT_EMPTY,
        accuracy_decimals=3,
        state_class=STATE_CLASS_MEASUREMENT,
    ),
    cv.Optional(CONF_IQR): sensor.sensor_schema(
        unit_of_measurement=UNIT_EMPTY,
        accuracy_decimals=3,
        state_class=STATE_CLASS_MEASUREMENT,
    ),
    cv.Optional(CONF_SPATIAL_VARIANCE): sensor.sensor_schema(
        unit_of_measurement=UNIT_EMPTY,
        accuracy_decimals=3,
        state_class=STATE_CLASS_MEASUREMENT,
    ),
    cv.Optional(CONF_SPATIAL_CORRELATION): sensor.sensor_schema(
        unit_of_measurement=UNIT_EMPTY,
        accuracy_decimals=3,
        state_class=STATE_CLASS_MEASUREMENT,
    ),
    cv.Optional(CONF_SPATIAL_GRADIENT): sensor.sensor_schema(
        unit_of_measurement=UNIT_EMPTY,
        accuracy_decimals=3,
        state_class=STATE_CLASS_MEASUREMENT,
    ),
    cv.Optional(CONF_TEMPORAL_DELTA_MEAN): sensor.sensor_schema(
        unit_of_measurement=UNIT_EMPTY,
        accuracy_decimals=3,
        state_class=STATE_CLASS_MEASUREMENT,
    ),
    cv.Optional(CONF_TEMPORAL_DELTA_VARIANCE): sensor.sensor_schema(
        unit_of_measurement=UNIT_EMPTY,
        accuracy_decimals=3,
        state_class=STATE_CLASS_MEASUREMENT,
    ),
})

async def to_code(config):
    parent = await cg.get_variable(config[CONF_ID])
    
    # Core sensors
    if CONF_MOVEMENT in config:
        sens = await sensor.new_sensor(config[CONF_MOVEMENT])
        cg.add(parent.set_movement_sensor(sens))
    
    if CONF_THRESHOLD in config:
        sens = await sensor.new_sensor(config[CONF_THRESHOLD])
        cg.add(parent.set_threshold_sensor(sens))
    
    if CONF_TURBULENCE in config:
        sens = await sensor.new_sensor(config[CONF_TURBULENCE])
        cg.add(parent.set_turbulence_sensor(sens))
    
    # Feature sensors
    if CONF_VARIANCE in config:
        sens = await sensor.new_sensor(config[CONF_VARIANCE])
        cg.add(parent.set_variance_sensor(sens))
    
    if CONF_SKEWNESS in config:
        sens = await sensor.new_sensor(config[CONF_SKEWNESS])
        cg.add(parent.set_skewness_sensor(sens))
    
    if CONF_KURTOSIS in config:
        sens = await sensor.new_sensor(config[CONF_KURTOSIS])
        cg.add(parent.set_kurtosis_sensor(sens))
    
    if CONF_ENTROPY in config:
        sens = await sensor.new_sensor(config[CONF_ENTROPY])
        cg.add(parent.set_entropy_sensor(sens))
    
    if CONF_IQR in config:
        sens = await sensor.new_sensor(config[CONF_IQR])
        cg.add(parent.set_iqr_sensor(sens))
    
    if CONF_SPATIAL_VARIANCE in config:
        sens = await sensor.new_sensor(config[CONF_SPATIAL_VARIANCE])
        cg.add(parent.set_spatial_variance_sensor(sens))
    
    if CONF_SPATIAL_CORRELATION in config:
        sens = await sensor.new_sensor(config[CONF_SPATIAL_CORRELATION])
        cg.add(parent.set_spatial_correlation_sensor(sens))
    
    if CONF_SPATIAL_GRADIENT in config:
        sens = await sensor.new_sensor(config[CONF_SPATIAL_GRADIENT])
        cg.add(parent.set_spatial_gradient_sensor(sens))
    
    if CONF_TEMPORAL_DELTA_MEAN in config:
        sens = await sensor.new_sensor(config[CONF_TEMPORAL_DELTA_MEAN])
        cg.add(parent.set_temporal_delta_mean_sensor(sens))
    
    if CONF_TEMPORAL_DELTA_VARIANCE in config:
        sens = await sensor.new_sensor(config[CONF_TEMPORAL_DELTA_VARIANCE])
        cg.add(parent.set_temporal_delta_variance_sensor(sens))
