// SPDX-License-Identifier: GPL-3.0-only
// Commercial licensing available under separate agreement; see LICENSING.md.

#include <stdint.h>
#include <string.h>

#include "native_features.h"
#include "py/objarray.h"
#include "py/runtime.h"

#define NATIVE_FEATURES_MAX_SUBCARRIERS (12)

typedef struct {
    mp_obj_base_t base;
    void *handle;
} native_detector_obj_t;

typedef struct {
    mp_obj_base_t base;
    void *handle;
} native_sampler_obj_t;

extern const mp_obj_type_t native_detector_type;
extern const mp_obj_type_t native_sampler_type;

static size_t native_get_subcarriers(
    mp_obj_t subcarriers_obj,
    uint8_t subcarriers[NATIVE_FEATURES_MAX_SUBCARRIERS]
) {
    size_t count;
    mp_obj_t *items;
    mp_obj_get_array(subcarriers_obj, &count, &items);
    if (count == 0 || count > NATIVE_FEATURES_MAX_SUBCARRIERS) {
        mp_raise_ValueError(MP_ERROR_TEXT("invalid subcarrier selection"));
    }
    for (size_t index = 0; index < count; ++index) {
        mp_int_t value = mp_obj_get_int(items[index]);
        if (value < 0 || value > 63) {
            mp_raise_ValueError(MP_ERROR_TEXT("invalid subcarrier index"));
        }
        subcarriers[index] = (uint8_t)value;
    }
    return count;
}

static native_detector_obj_t *native_detector_get(mp_obj_t self_in) {
    if (!mp_obj_is_type(self_in, &native_detector_type)) {
        mp_raise_TypeError(MP_ERROR_TEXT("expected Detector"));
    }
    native_detector_obj_t *self = MP_OBJ_TO_PTR(self_in);
    if (self->handle == NULL) {
        mp_raise_ValueError(MP_ERROR_TEXT("detector is deinitialized"));
    }
    return self;
}

static mp_obj_t native_detector_make_new(
    const mp_obj_type_t *type,
    size_t n_args,
    size_t n_kw,
    const mp_obj_t *all_args
) {
    enum {
        ARG_algorithm,
        ARG_window_size,
        ARG_threshold,
        ARG_lag,
        ARG_enable_hampel,
        ARG_hampel_window,
        ARG_hampel_threshold,
        ARG_enable_lowpass,
        ARG_lowpass_cutoff,
        ARG_subcarriers,
    };
    static const mp_arg_t allowed_args[] = {
        {MP_QSTR_algorithm, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_obj = mp_const_none}},
        {MP_QSTR_window_size, MP_ARG_REQUIRED | MP_ARG_KW_ONLY | MP_ARG_INT, {.u_int = 0}},
        {MP_QSTR_threshold, MP_ARG_REQUIRED | MP_ARG_KW_ONLY | MP_ARG_OBJ, {.u_obj = mp_const_none}},
        {MP_QSTR_lag, MP_ARG_REQUIRED | MP_ARG_KW_ONLY | MP_ARG_INT, {.u_int = 0}},
        {MP_QSTR_enable_hampel, MP_ARG_KW_ONLY | MP_ARG_BOOL, {.u_bool = true}},
        {MP_QSTR_hampel_window, MP_ARG_KW_ONLY | MP_ARG_INT, {.u_int = 7}},
        {MP_QSTR_hampel_threshold, MP_ARG_KW_ONLY | MP_ARG_OBJ, {.u_obj = mp_const_none}},
        {MP_QSTR_enable_lowpass, MP_ARG_KW_ONLY | MP_ARG_BOOL, {.u_bool = false}},
        {MP_QSTR_lowpass_cutoff, MP_ARG_KW_ONLY | MP_ARG_OBJ, {.u_obj = mp_const_none}},
        {MP_QSTR_subcarriers, MP_ARG_REQUIRED | MP_ARG_KW_ONLY | MP_ARG_OBJ, {.u_obj = mp_const_none}},
    };
    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all_kw_array(
        n_args,
        n_kw,
        all_args,
        MP_ARRAY_SIZE(allowed_args),
        allowed_args,
        args
    );

    const char *algorithm = mp_obj_str_get_str(args[ARG_algorithm].u_obj);
    espectre_native_detector_kind_t kind;
    if (strcmp(algorithm, "lightweight") == 0) {
        kind = ESPECTRE_NATIVE_DETECTOR_LIGHTWEIGHT;
    } else {
        mp_raise_ValueError(MP_ERROR_TEXT("unsupported detector algorithm"));
    }
    mp_int_t window_size = args[ARG_window_size].u_int;
    mp_int_t lag = args[ARG_lag].u_int;
    mp_int_t hampel_window = args[ARG_hampel_window].u_int;
    if (window_size <= 0 || window_size > UINT16_MAX ||
        lag <= 0 || lag > UINT16_MAX ||
        hampel_window < 3 || hampel_window > 11) {
        mp_raise_ValueError(MP_ERROR_TEXT("invalid detector configuration"));
    }
    uint8_t subcarriers[NATIVE_FEATURES_MAX_SUBCARRIERS];
    size_t subcarrier_count = native_get_subcarriers(
        args[ARG_subcarriers].u_obj,
        subcarriers
    );
    mp_float_t hampel_threshold =
        args[ARG_hampel_threshold].u_obj == mp_const_none
            ? 5.0f
            : mp_obj_get_float(args[ARG_hampel_threshold].u_obj);
    mp_float_t lowpass_cutoff =
        args[ARG_lowpass_cutoff].u_obj == mp_const_none
            ? 11.0f
            : mp_obj_get_float(args[ARG_lowpass_cutoff].u_obj);

    native_detector_obj_t *self = mp_obj_malloc_with_finaliser(
        native_detector_obj_t,
        type
    );
    self->handle = NULL;
    self->handle = espectre_native_detector_create(
        kind,
        (uint16_t)window_size,
        mp_obj_get_float(args[ARG_threshold].u_obj),
        (uint16_t)lag,
        args[ARG_enable_hampel].u_bool,
        (uint8_t)hampel_window,
        hampel_threshold,
        args[ARG_enable_lowpass].u_bool,
        lowpass_cutoff,
        subcarriers,
        (uint8_t)subcarrier_count
    );
    if (self->handle == NULL) {
        mp_raise_msg(&mp_type_MemoryError, MP_ERROR_TEXT("detector allocation failed"));
    }
    return MP_OBJ_FROM_PTR(self);
}

static mp_obj_t native_detector_deinit(mp_obj_t self_in) {
    native_detector_obj_t *self = MP_OBJ_TO_PTR(self_in);
    if (self->handle != NULL) {
        espectre_native_detector_destroy(self->handle);
        self->handle = NULL;
    }
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(native_detector_deinit_obj, native_detector_deinit);

static mp_obj_t native_detector_process(size_t n_args, const mp_obj_t *args) {
    native_detector_obj_t *self = native_detector_get(args[0]);
    mp_buffer_info_t csi;
    mp_get_buffer_raise(args[1], &csi, MP_BUFFER_READ);
    uint32_t timestamp_us = n_args > 2
        ? (uint32_t)mp_obj_get_int_truncated(args[2])
        : 0U;
    if (!espectre_native_detector_process(
            self->handle,
            (const int8_t *)csi.buf,
            csi.len,
            timestamp_us)) {
        mp_raise_ValueError(MP_ERROR_TEXT("detector process failed"));
    }
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(
    native_detector_process_obj,
    2,
    3,
    native_detector_process
);

static mp_obj_t native_detector_update(mp_obj_t self_in, mp_obj_t output_in) {
    native_detector_obj_t *self = native_detector_get(self_in);
    mp_buffer_info_t output;
    mp_get_buffer_raise(output_in, &output, MP_BUFFER_WRITE);
    if (output.typecode != 'f' || output.len < 6 * sizeof(float)) {
        mp_raise_ValueError(MP_ERROR_TEXT("expected six-float output array"));
    }
    if (!espectre_native_detector_update(self->handle, (float *)output.buf)) {
        mp_raise_ValueError(MP_ERROR_TEXT("detector update failed"));
    }
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(native_detector_update_obj, native_detector_update);

static mp_obj_t native_detector_set_subcarriers(
    mp_obj_t self_in,
    mp_obj_t subcarriers_in
) {
    native_detector_obj_t *self = native_detector_get(self_in);
    uint8_t subcarriers[NATIVE_FEATURES_MAX_SUBCARRIERS];
    size_t count = native_get_subcarriers(subcarriers_in, subcarriers);
    if (!espectre_native_detector_set_subcarriers(
            self->handle,
            subcarriers,
            (uint8_t)count)) {
        mp_raise_ValueError(MP_ERROR_TEXT("invalid subcarrier selection"));
    }
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(
    native_detector_set_subcarriers_obj,
    native_detector_set_subcarriers
);

static mp_obj_t native_detector_advance_missing(
    mp_obj_t self_in,
    mp_obj_t value_in
) {
    native_detector_obj_t *self = native_detector_get(self_in);
    mp_int_t value = mp_obj_get_int(value_in);
    if (value < 0 || !espectre_native_detector_advance_missing(
            self->handle,
            (uint32_t)value)) {
        mp_raise_ValueError(MP_ERROR_TEXT("invalid missing-slot count"));
    }
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(
    native_detector_advance_missing_obj,
    native_detector_advance_missing
);

static mp_obj_t native_detector_set_minimum_valid(
    mp_obj_t self_in,
    mp_obj_t value_in
) {
    native_detector_obj_t *self = native_detector_get(self_in);
    mp_int_t value = mp_obj_get_int(value_in);
    if (value < 0 || value > UINT16_MAX ||
        !espectre_native_detector_set_minimum_valid(
            self->handle,
            (uint16_t)value)) {
        mp_raise_ValueError(MP_ERROR_TEXT("invalid minimum-valid count"));
    }
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(
    native_detector_set_minimum_valid_obj,
    native_detector_set_minimum_valid
);

static mp_obj_t native_detector_reset(mp_obj_t self_in) {
    native_detector_obj_t *self = native_detector_get(self_in);
    if (!espectre_native_detector_reset(self->handle)) {
        mp_raise_ValueError(MP_ERROR_TEXT("detector reset failed"));
    }
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(native_detector_reset_obj, native_detector_reset);

static mp_obj_t native_detector_is_ready(mp_obj_t self_in) {
    native_detector_obj_t *self = native_detector_get(self_in);
    return mp_obj_new_bool(espectre_native_detector_is_ready(self->handle));
}
static MP_DEFINE_CONST_FUN_OBJ_1(native_detector_is_ready_obj, native_detector_is_ready);

static mp_obj_t native_detector_set_threshold(mp_obj_t self_in, mp_obj_t value_in) {
    native_detector_obj_t *self = native_detector_get(self_in);
    if (!espectre_native_detector_set_threshold(
            self->handle,
            mp_obj_get_float(value_in))) {
        mp_raise_ValueError(MP_ERROR_TEXT("invalid threshold"));
    }
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(
    native_detector_set_threshold_obj,
    native_detector_set_threshold
);

static mp_obj_t native_detector_get_threshold(mp_obj_t self_in) {
    native_detector_obj_t *self = native_detector_get(self_in);
    return mp_obj_new_float(espectre_native_detector_get_threshold(self->handle));
}
static MP_DEFINE_CONST_FUN_OBJ_1(
    native_detector_get_threshold_obj,
    native_detector_get_threshold
);

static mp_obj_t native_detector_get_metric(mp_obj_t self_in) {
    native_detector_obj_t *self = native_detector_get(self_in);
    return mp_obj_new_float(espectre_native_detector_get_metric(self->handle));
}
static MP_DEFINE_CONST_FUN_OBJ_1(
    native_detector_get_metric_obj,
    native_detector_get_metric
);

static mp_obj_t native_detector_get_total_packets(mp_obj_t self_in) {
    native_detector_obj_t *self = native_detector_get(self_in);
    return mp_obj_new_int_from_uint(
        espectre_native_detector_get_total_packets(self->handle)
    );
}
static MP_DEFINE_CONST_FUN_OBJ_1(
    native_detector_get_total_packets_obj,
    native_detector_get_total_packets
);

static mp_obj_t native_detector_calibration_begin(mp_obj_t self_in) {
    native_detector_obj_t *self = native_detector_get(self_in);
    espectre_native_detector_calibration_begin(self->handle);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(
    native_detector_calibration_begin_obj,
    native_detector_calibration_begin
);

static mp_obj_t native_detector_calibration_complete(mp_obj_t self_in) {
    native_detector_obj_t *self = native_detector_get(self_in);
    espectre_native_detector_calibration_complete(self->handle);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(
    native_detector_calibration_complete_obj,
    native_detector_calibration_complete
);

static mp_obj_t native_detector_apply_adaptive_threshold(mp_obj_t self_in) {
    native_detector_obj_t *self = native_detector_get(self_in);
    if (!espectre_native_detector_apply_adaptive_threshold(self->handle, 0.0f)) {
        mp_raise_ValueError(MP_ERROR_TEXT("adaptive threshold failed"));
    }
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(
    native_detector_apply_adaptive_threshold_obj,
    native_detector_apply_adaptive_threshold
);

static const mp_rom_map_elem_t native_detector_locals_table[] = {
    {MP_ROM_QSTR(MP_QSTR___del__), MP_ROM_PTR(&native_detector_deinit_obj)},
    {MP_ROM_QSTR(MP_QSTR_deinit), MP_ROM_PTR(&native_detector_deinit_obj)},
    {MP_ROM_QSTR(MP_QSTR_process), MP_ROM_PTR(&native_detector_process_obj)},
    {MP_ROM_QSTR(MP_QSTR_update), MP_ROM_PTR(&native_detector_update_obj)},
    {MP_ROM_QSTR(MP_QSTR_set_subcarriers), MP_ROM_PTR(&native_detector_set_subcarriers_obj)},
    {MP_ROM_QSTR(MP_QSTR_advance_missing), MP_ROM_PTR(&native_detector_advance_missing_obj)},
    {MP_ROM_QSTR(MP_QSTR_set_minimum_valid), MP_ROM_PTR(&native_detector_set_minimum_valid_obj)},
    {MP_ROM_QSTR(MP_QSTR_reset), MP_ROM_PTR(&native_detector_reset_obj)},
    {MP_ROM_QSTR(MP_QSTR_is_ready), MP_ROM_PTR(&native_detector_is_ready_obj)},
    {MP_ROM_QSTR(MP_QSTR_set_threshold), MP_ROM_PTR(&native_detector_set_threshold_obj)},
    {MP_ROM_QSTR(MP_QSTR_get_threshold), MP_ROM_PTR(&native_detector_get_threshold_obj)},
    {MP_ROM_QSTR(MP_QSTR_get_metric), MP_ROM_PTR(&native_detector_get_metric_obj)},
    {MP_ROM_QSTR(MP_QSTR_get_total_packets), MP_ROM_PTR(&native_detector_get_total_packets_obj)},
    {MP_ROM_QSTR(MP_QSTR_calibration_begin), MP_ROM_PTR(&native_detector_calibration_begin_obj)},
    {MP_ROM_QSTR(MP_QSTR_calibration_complete), MP_ROM_PTR(&native_detector_calibration_complete_obj)},
    {MP_ROM_QSTR(MP_QSTR_apply_adaptive_threshold), MP_ROM_PTR(&native_detector_apply_adaptive_threshold_obj)},
};
static MP_DEFINE_CONST_DICT(native_detector_locals, native_detector_locals_table);

MP_DEFINE_CONST_OBJ_TYPE(
    native_detector_type,
    MP_QSTR_Detector,
    MP_TYPE_FLAG_NONE,
    make_new, native_detector_make_new,
    locals_dict, &native_detector_locals
);

static native_sampler_obj_t *native_sampler_get(mp_obj_t self_in) {
    if (!mp_obj_is_type(self_in, &native_sampler_type)) {
        mp_raise_TypeError(MP_ERROR_TEXT("expected TemporalCsiSampler"));
    }
    native_sampler_obj_t *self = MP_OBJ_TO_PTR(self_in);
    if (self->handle == NULL) {
        mp_raise_ValueError(MP_ERROR_TEXT("sampler is deinitialized"));
    }
    return self;
}

static mp_obj_t native_sampler_make_new(
    const mp_obj_type_t *type,
    size_t n_args,
    size_t n_kw,
    const mp_obj_t *all_args
) {
    enum {ARG_target_pps, ARG_window_size_ms};
    static const mp_arg_t allowed_args[] = {
        {MP_QSTR_target_pps, MP_ARG_REQUIRED | MP_ARG_INT, {.u_int = 0}},
        {MP_QSTR_window_size_ms, MP_ARG_REQUIRED | MP_ARG_INT, {.u_int = 0}},
    };
    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all_kw_array(
        n_args,
        n_kw,
        all_args,
        MP_ARRAY_SIZE(allowed_args),
        allowed_args,
        args
    );
    if (args[ARG_target_pps].u_int <= 0 || args[ARG_window_size_ms].u_int <= 0) {
        mp_raise_ValueError(MP_ERROR_TEXT("invalid sampler configuration"));
    }
    native_sampler_obj_t *self = mp_obj_malloc_with_finaliser(
        native_sampler_obj_t,
        type
    );
    self->handle = espectre_native_sampler_create(
        (uint32_t)args[ARG_target_pps].u_int,
        (uint32_t)args[ARG_window_size_ms].u_int
    );
    if (self->handle == NULL) {
        mp_raise_msg(&mp_type_MemoryError, MP_ERROR_TEXT("sampler allocation failed"));
    }
    return MP_OBJ_FROM_PTR(self);
}

static mp_obj_t native_sampler_deinit(mp_obj_t self_in) {
    native_sampler_obj_t *self = MP_OBJ_TO_PTR(self_in);
    if (self->handle != NULL) {
        espectre_native_sampler_destroy(self->handle);
        self->handle = NULL;
    }
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(native_sampler_deinit_obj, native_sampler_deinit);

static mp_obj_t native_sampler_reset(mp_obj_t self_in) {
    native_sampler_obj_t *self = native_sampler_get(self_in);
    espectre_native_sampler_reset(self->handle);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(native_sampler_reset_obj, native_sampler_reset);

static mp_obj_t native_sampler_clear_history(mp_obj_t self_in) {
    native_sampler_obj_t *self = native_sampler_get(self_in);
    espectre_native_sampler_clear_history(self->handle);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(
    native_sampler_clear_history_obj,
    native_sampler_clear_history
);

static mp_obj_t native_sampler_clear_window_preserving_phase(mp_obj_t self_in) {
    native_sampler_obj_t *self = native_sampler_get(self_in);
    espectre_native_sampler_clear_window_preserving_phase(self->handle);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(
    native_sampler_clear_window_preserving_phase_obj,
    native_sampler_clear_window_preserving_phase
);

static mp_obj_t native_sampler_admit(size_t n_args, const mp_obj_t *args) {
    native_sampler_obj_t *self = native_sampler_get(args[0]);
    bool has_timestamp = args[1] != mp_const_none;
    uint32_t timestamp_us = has_timestamp
        ? (uint32_t)mp_obj_get_int_truncated(args[1])
        : 0U;
    bool has_now = n_args > 2 && args[2] != mp_const_none;
    uint32_t now_us = has_now ? (uint32_t)mp_obj_get_int_truncated(args[2]) : 0U;
    return mp_obj_new_bool(espectre_native_sampler_admit(
        self->handle,
        timestamp_us,
        has_timestamp,
        now_us,
        has_now
    ));
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(
    native_sampler_admit_obj,
    2,
    3,
    native_sampler_admit
);

static mp_obj_t native_sampler_flush(mp_obj_t self_in) {
    native_sampler_obj_t *self = native_sampler_get(self_in);
    return mp_obj_new_bool(espectre_native_sampler_flush(self->handle));
}
static MP_DEFINE_CONST_FUN_OBJ_1(native_sampler_flush_obj, native_sampler_flush);

static mp_obj_t native_sampler_get_u32(mp_obj_t self_in, mp_obj_t field_in) {
    native_sampler_obj_t *self = native_sampler_get(self_in);
    return mp_obj_new_int_from_uint(espectre_native_sampler_get_u32(
        self->handle,
        (uint8_t)mp_obj_get_int(field_in)
    ));
}
static MP_DEFINE_CONST_FUN_OBJ_2(native_sampler_get_u32_obj, native_sampler_get_u32);

static mp_obj_t native_sampler_get_u64(mp_obj_t self_in, mp_obj_t field_in) {
    native_sampler_obj_t *self = native_sampler_get(self_in);
    return mp_obj_new_int_from_ull(espectre_native_sampler_get_u64(
        self->handle,
        (uint8_t)mp_obj_get_int(field_in)
    ));
}
static MP_DEFINE_CONST_FUN_OBJ_2(native_sampler_get_u64_obj, native_sampler_get_u64);

static mp_obj_t native_sampler_get_flag(mp_obj_t self_in, mp_obj_t field_in) {
    native_sampler_obj_t *self = native_sampler_get(self_in);
    return mp_obj_new_bool(espectre_native_sampler_get_flag(
        self->handle,
        (uint8_t)mp_obj_get_int(field_in)
    ));
}
static MP_DEFINE_CONST_FUN_OBJ_2(native_sampler_get_flag_obj, native_sampler_get_flag);

static mp_obj_t native_sampler_occupancy_ratio(mp_obj_t self_in) {
    native_sampler_obj_t *self = native_sampler_get(self_in);
    return mp_obj_new_float(espectre_native_sampler_get_occupancy_ratio(self->handle));
}
static MP_DEFINE_CONST_FUN_OBJ_1(
    native_sampler_occupancy_ratio_obj,
    native_sampler_occupancy_ratio
);

static mp_obj_t native_sampler_snapshot(mp_obj_t self_in) {
    native_sampler_obj_t *self = native_sampler_get(self_in);
    mp_obj_t values[24] = {
        mp_obj_new_int_from_uint(espectre_native_sampler_get_u32(self->handle, 0)),
        mp_obj_new_int_from_uint(espectre_native_sampler_get_u32(self->handle, 1)),
        mp_obj_new_int_from_uint(espectre_native_sampler_get_u32(self->handle, 2)),
        mp_obj_new_int_from_uint(espectre_native_sampler_get_u32(self->handle, 3)),
        mp_obj_new_int_from_uint(espectre_native_sampler_get_u32(self->handle, 4)),
        mp_obj_new_int_from_uint(espectre_native_sampler_get_u32(self->handle, 5)),
        mp_obj_new_float(espectre_native_sampler_get_occupancy_ratio(self->handle)),
        mp_obj_new_bool(espectre_native_sampler_get_flag(self->handle, 0)),
        mp_obj_new_bool(espectre_native_sampler_get_flag(self->handle, 1)),
        mp_obj_new_bool(espectre_native_sampler_get_flag(self->handle, 2)),
        mp_obj_new_bool(espectre_native_sampler_get_flag(self->handle, 3)),
        mp_obj_new_bool(espectre_native_sampler_get_flag(self->handle, 4)),
        mp_obj_new_bool(espectre_native_sampler_get_flag(self->handle, 5)),
        mp_obj_new_int_from_ull(espectre_native_sampler_get_u64(self->handle, 0)),
        mp_obj_new_int_from_ull(espectre_native_sampler_get_u64(self->handle, 1)),
        mp_obj_new_int_from_ull(espectre_native_sampler_get_u64(self->handle, 2)),
        mp_obj_new_int_from_ull(espectre_native_sampler_get_u64(self->handle, 3)),
        mp_obj_new_int_from_ull(espectre_native_sampler_get_u64(self->handle, 4)),
        mp_obj_new_int_from_ull(espectre_native_sampler_get_u64(self->handle, 5)),
        mp_obj_new_int_from_ull(espectre_native_sampler_get_u64(self->handle, 6)),
        mp_obj_new_int_from_ull(espectre_native_sampler_get_u64(self->handle, 7)),
        mp_obj_new_int_from_ull(espectre_native_sampler_get_u64(self->handle, 8)),
        mp_obj_new_int_from_ull(espectre_native_sampler_get_u64(self->handle, 9)),
        mp_obj_new_int_from_ull(espectre_native_sampler_get_u64(self->handle, 10)),
    };
    return mp_obj_new_tuple(MP_ARRAY_SIZE(values), values);
}
static MP_DEFINE_CONST_FUN_OBJ_1(native_sampler_snapshot_obj, native_sampler_snapshot);

static const mp_rom_map_elem_t native_sampler_locals_table[] = {
    {MP_ROM_QSTR(MP_QSTR___del__), MP_ROM_PTR(&native_sampler_deinit_obj)},
    {MP_ROM_QSTR(MP_QSTR_deinit), MP_ROM_PTR(&native_sampler_deinit_obj)},
    {MP_ROM_QSTR(MP_QSTR_reset), MP_ROM_PTR(&native_sampler_reset_obj)},
    {MP_ROM_QSTR(MP_QSTR_clear_history), MP_ROM_PTR(&native_sampler_clear_history_obj)},
    {MP_ROM_QSTR(MP_QSTR_clear_window_preserving_phase), MP_ROM_PTR(&native_sampler_clear_window_preserving_phase_obj)},
    {MP_ROM_QSTR(MP_QSTR_admit), MP_ROM_PTR(&native_sampler_admit_obj)},
    {MP_ROM_QSTR(MP_QSTR_flush), MP_ROM_PTR(&native_sampler_flush_obj)},
    {MP_ROM_QSTR(MP_QSTR_get_u32), MP_ROM_PTR(&native_sampler_get_u32_obj)},
    {MP_ROM_QSTR(MP_QSTR_get_u64), MP_ROM_PTR(&native_sampler_get_u64_obj)},
    {MP_ROM_QSTR(MP_QSTR_get_flag), MP_ROM_PTR(&native_sampler_get_flag_obj)},
    {MP_ROM_QSTR(MP_QSTR_occupancy_ratio), MP_ROM_PTR(&native_sampler_occupancy_ratio_obj)},
    {MP_ROM_QSTR(MP_QSTR_snapshot), MP_ROM_PTR(&native_sampler_snapshot_obj)},
};
static MP_DEFINE_CONST_DICT(native_sampler_locals, native_sampler_locals_table);

MP_DEFINE_CONST_OBJ_TYPE(
    native_sampler_type,
    MP_QSTR_TemporalCsiSampler,
    MP_TYPE_FLAG_NONE,
    make_new, native_sampler_make_new,
    locals_dict, &native_sampler_locals
);

static const mp_rom_map_elem_t native_features_module_globals_table[] = {
    {MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_espectre_native_features)},
    {MP_ROM_QSTR(MP_QSTR_BACKEND), MP_ROM_QSTR(MP_QSTR_espectre_core)},
    {MP_ROM_QSTR(MP_QSTR_Detector), MP_ROM_PTR(&native_detector_type)},
    {MP_ROM_QSTR(MP_QSTR_TemporalCsiSampler), MP_ROM_PTR(&native_sampler_type)},
};
static MP_DEFINE_CONST_DICT(
    native_features_module_globals,
    native_features_module_globals_table
);

const mp_obj_module_t native_features_module = {
    .base = {&mp_type_module},
    .globals = (mp_obj_dict_t *)&native_features_module_globals,
};

MP_REGISTER_MODULE(MP_QSTR_espectre_native_features, native_features_module);
