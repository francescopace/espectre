// SPDX-License-Identifier: GPL-3.0-only
// Commercial licensing available under separate agreement; see LICENSING.md.

#include "native_wifi.h"
#include "py/runtime.h"

static mp_obj_t native_wifi_prepare_tx_rate(void) {
  const esp_err_t err = espectre_native_wifi_prepare_tx_rate();
  if (err != ESP_OK) {
    mp_raise_OSError(err);
  }
  return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_0(native_wifi_prepare_tx_rate_obj, native_wifi_prepare_tx_rate);

static mp_obj_t native_wifi_apply_tx_rate(void) {
  const esp_err_t err = espectre_native_wifi_apply_tx_rate();
  if (err != ESP_OK) {
    mp_raise_OSError(err);
  }
  return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_0(native_wifi_apply_tx_rate_obj, native_wifi_apply_tx_rate);

static const mp_rom_map_elem_t native_wifi_module_globals_table[] = {
    {MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_espectre_native_wifi)},
    {MP_ROM_QSTR(MP_QSTR_prepare_tx_rate), MP_ROM_PTR(&native_wifi_prepare_tx_rate_obj)},
    {MP_ROM_QSTR(MP_QSTR_apply_tx_rate), MP_ROM_PTR(&native_wifi_apply_tx_rate_obj)},
};
static MP_DEFINE_CONST_DICT(native_wifi_module_globals, native_wifi_module_globals_table);

const mp_obj_module_t native_wifi_module = {
    .base = {&mp_type_module},
    .globals = (mp_obj_dict_t *) &native_wifi_module_globals,
};

MP_REGISTER_MODULE(MP_QSTR_espectre_native_wifi, native_wifi_module);
