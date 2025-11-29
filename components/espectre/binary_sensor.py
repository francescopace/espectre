import esphome.codegen as cg
import esphome.config_validation as cv
from esphome.components import binary_sensor
from esphome.const import (
    CONF_ID,
    DEVICE_CLASS_MOTION,
)
from . import (
    espectre_ns,
    ESpectreComponent,
    CONF_MOTION,
)

DEPENDENCIES = ["espectre"]

CONFIG_SCHEMA = cv.Schema({
    cv.GenerateID(): cv.use_id(ESpectreComponent),
    
    cv.Optional(CONF_MOTION): binary_sensor.binary_sensor_schema(
        device_class=DEVICE_CLASS_MOTION,
    ),
})

async def to_code(config):
    parent = await cg.get_variable(config[CONF_ID])
    
    if CONF_MOTION in config:
        sens = await binary_sensor.new_binary_sensor(config[CONF_MOTION])
        cg.add(parent.set_motion_binary_sensor(sens))
