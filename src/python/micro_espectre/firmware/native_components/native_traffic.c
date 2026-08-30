// SPDX-License-Identifier: GPL-3.0-only
// Commercial licensing available under separate agreement; see LICENSING.md.

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "lwip/inet.h"
#include "lwip/sockets.h"
#include "native_traffic.h"
#include "py/mperrno.h"
#include "py/runtime.h"

typedef struct _native_traffic_obj_t {
  mp_obj_base_t base;
  void *handle;
} native_traffic_obj_t;

extern const mp_obj_type_t native_traffic_type;

static native_traffic_obj_t *native_traffic_get(mp_obj_t self_in) {
  if (!mp_obj_is_type(self_in, &native_traffic_type)) {
    mp_raise_TypeError(MP_ERROR_TEXT("expected TrafficGenerator"));
  }
  native_traffic_obj_t *self = MP_OBJ_TO_PTR(self_in);
  if (self->handle == NULL) {
    mp_raise_ValueError(MP_ERROR_TEXT("traffic generator is deinitialized"));
  }
  return self;
}

static espectre_native_traffic_mode_t native_traffic_mode(mp_obj_t mode_in) {
  const char *mode = mp_obj_str_get_str(mode_in);
  if (strcmp(mode, "ping") == 0) {
    return ESPECTRE_NATIVE_TRAFFIC_PING;
  }
  if (strcmp(mode, "dns") == 0) {
    return ESPECTRE_NATIVE_TRAFFIC_DNS;
  }
  if (strcmp(mode, "dns_tcp") == 0) {
    return ESPECTRE_NATIVE_TRAFFIC_DNS_TCP;
  }
  mp_raise_ValueError(MP_ERROR_TEXT("invalid traffic generator mode"));
}

static mp_obj_t native_traffic_make_new(
    const mp_obj_type_t *type,
    size_t n_args,
    size_t n_kw,
    const mp_obj_t *args) {
  mp_arg_check_num(n_args, n_kw, 0, 0, false);
  native_traffic_obj_t *self = mp_obj_malloc_with_finaliser(
      native_traffic_obj_t,
      type);
  self->handle = espectre_native_traffic_create();
  if (self->handle == NULL) {
    mp_raise_msg(
        &mp_type_MemoryError,
        MP_ERROR_TEXT("traffic generator allocation failed"));
  }
  return MP_OBJ_FROM_PTR(self);
}

static mp_obj_t native_traffic_start(size_t n_args, const mp_obj_t *args) {
  native_traffic_obj_t *self = native_traffic_get(args[0]);
  const char *gateway = mp_obj_str_get_str(args[1]);
  struct in_addr address;
  if (inet_pton(AF_INET, gateway, &address) != 1) {
    mp_raise_ValueError(MP_ERROR_TEXT("invalid gateway IPv4 address"));
  }
  mp_int_t rate = mp_obj_get_int(args[2]);
  if (rate <= 0 || rate > 1000) {
    mp_raise_ValueError(MP_ERROR_TEXT("rate must be 1..1000"));
  }
  espectre_native_traffic_mode_t mode = n_args > 3
      ? native_traffic_mode(args[3])
      : ESPECTRE_NATIVE_TRAFFIC_PING;
  if (!espectre_native_traffic_start(
          self->handle,
          address.s_addr,
          (uint32_t) rate,
          mode)) {
    mp_raise_OSError(MP_EIO);
  }
  return mp_const_true;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(
    native_traffic_start_obj,
    3,
    4,
    native_traffic_start);

static mp_obj_t native_traffic_stop(mp_obj_t self_in) {
  native_traffic_obj_t *self = native_traffic_get(self_in);
  espectre_native_traffic_stop(self->handle);
  return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(native_traffic_stop_obj, native_traffic_stop);

static mp_obj_t native_traffic_deinit(mp_obj_t self_in) {
  native_traffic_obj_t *self = MP_OBJ_TO_PTR(self_in);
  if (self->handle != NULL) {
    espectre_native_traffic_destroy(self->handle);
    self->handle = NULL;
  }
  return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(native_traffic_deinit_obj, native_traffic_deinit);

static mp_obj_t native_traffic_pause(mp_obj_t self_in) {
  native_traffic_obj_t *self = native_traffic_get(self_in);
  return mp_obj_new_bool(espectre_native_traffic_pause(self->handle));
}
static MP_DEFINE_CONST_FUN_OBJ_1(native_traffic_pause_obj, native_traffic_pause);

static mp_obj_t native_traffic_resume(mp_obj_t self_in) {
  native_traffic_obj_t *self = native_traffic_get(self_in);
  return mp_obj_new_bool(espectre_native_traffic_resume(self->handle));
}
static MP_DEFINE_CONST_FUN_OBJ_1(native_traffic_resume_obj, native_traffic_resume);

static mp_obj_t native_traffic_is_running(mp_obj_t self_in) {
  native_traffic_obj_t *self = native_traffic_get(self_in);
  return mp_obj_new_bool(espectre_native_traffic_is_running(self->handle));
}
static MP_DEFINE_CONST_FUN_OBJ_1(native_traffic_is_running_obj, native_traffic_is_running);

static mp_obj_t native_traffic_packet_count(mp_obj_t self_in) {
  native_traffic_obj_t *self = native_traffic_get(self_in);
  return mp_obj_new_int_from_uint(
      espectre_native_traffic_packet_count(self->handle));
}
static MP_DEFINE_CONST_FUN_OBJ_1(native_traffic_packet_count_obj, native_traffic_packet_count);

static mp_obj_t native_traffic_error_count(mp_obj_t self_in) {
  native_traffic_obj_t *self = native_traffic_get(self_in);
  return mp_obj_new_int_from_uint(
      espectre_native_traffic_error_count(self->handle));
}
static MP_DEFINE_CONST_FUN_OBJ_1(native_traffic_error_count_obj, native_traffic_error_count);

static const mp_rom_map_elem_t native_traffic_locals_table[] = {
    {MP_ROM_QSTR(MP_QSTR_start), MP_ROM_PTR(&native_traffic_start_obj)},
    {MP_ROM_QSTR(MP_QSTR_stop), MP_ROM_PTR(&native_traffic_stop_obj)},
    {MP_ROM_QSTR(MP_QSTR_pause), MP_ROM_PTR(&native_traffic_pause_obj)},
    {MP_ROM_QSTR(MP_QSTR_resume), MP_ROM_PTR(&native_traffic_resume_obj)},
    {MP_ROM_QSTR(MP_QSTR_is_running), MP_ROM_PTR(&native_traffic_is_running_obj)},
    {MP_ROM_QSTR(MP_QSTR_packet_count), MP_ROM_PTR(&native_traffic_packet_count_obj)},
    {MP_ROM_QSTR(MP_QSTR_error_count), MP_ROM_PTR(&native_traffic_error_count_obj)},
    {MP_ROM_QSTR(MP_QSTR_deinit), MP_ROM_PTR(&native_traffic_deinit_obj)},
    {MP_ROM_QSTR(MP_QSTR___del__), MP_ROM_PTR(&native_traffic_deinit_obj)},
};
static MP_DEFINE_CONST_DICT(native_traffic_locals, native_traffic_locals_table);

MP_DEFINE_CONST_OBJ_TYPE(
    native_traffic_type,
    MP_QSTR_TrafficGenerator,
    MP_TYPE_FLAG_NONE,
    make_new, native_traffic_make_new,
    locals_dict, &native_traffic_locals);

static const mp_rom_map_elem_t native_traffic_module_globals_table[] = {
    {MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_espectre_native_traffic)},
    {MP_ROM_QSTR(MP_QSTR_TrafficGenerator), MP_ROM_PTR(&native_traffic_type)},
    // Keep the old constructor name compatible with deployed application bytecode.
    {MP_ROM_QSTR(MP_QSTR_PingGenerator), MP_ROM_PTR(&native_traffic_type)},
};
static MP_DEFINE_CONST_DICT(native_traffic_module_globals, native_traffic_module_globals_table);

const mp_obj_module_t native_traffic_module = {
    .base = {&mp_type_module},
    .globals = (mp_obj_dict_t *) &native_traffic_module_globals,
};

MP_REGISTER_MODULE(MP_QSTR_espectre_native_traffic, native_traffic_module);
