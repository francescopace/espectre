// SPDX-License-Identifier: GPL-3.0-only
// Commercial licensing available under separate agreement; see LICENSING.md.

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "esp_err.h"
#include "mqtt_client.h"
#include "freertos/FreeRTOS.h"
#include "freertos/portmacro.h"
#include "py/mperrno.h"
#include "py/mphal.h"
#include "py/runtime.h"

#define NATIVE_MQTT_SUBSCRIBE_TIMEOUT_MS (5000)
#define NATIVE_MQTT_TASK_STACK_SIZE (3072)
#define NATIVE_MQTT_BUFFER_SIZE (512)
#define NATIVE_MQTT_TOPIC_SIZE (128)
#define NATIVE_MQTT_PAYLOAD_SIZE (256)

typedef enum {
    NATIVE_MQTT_IDLE = 0,
    NATIVE_MQTT_CONNECTING,
    NATIVE_MQTT_CONNECTED,
    NATIVE_MQTT_FAILED,
} native_mqtt_state_t;

typedef struct _native_mqtt_client_obj_t {
    mp_obj_base_t base;
    esp_mqtt_client_handle_t client;
    mp_obj_t callback;
    volatile native_mqtt_state_t state;
    volatile int subscribed_msg_id;
    bool started;
    volatile bool message_pending;
    portMUX_TYPE message_lock;
    size_t topic_len;
    size_t payload_len;
    char topic[NATIVE_MQTT_TOPIC_SIZE];
    uint8_t payload[NATIVE_MQTT_PAYLOAD_SIZE];
} native_mqtt_client_obj_t;

static void native_mqtt_event_handler(
    void *handler_args,
    esp_event_base_t event_base,
    int32_t event_id,
    void *event_data
) {
    (void)event_base;
    native_mqtt_client_obj_t *self = handler_args;
    esp_mqtt_event_handle_t event = event_data;

    switch ((esp_mqtt_event_id_t)event_id) {
        case MQTT_EVENT_CONNECTED:
            self->state = NATIVE_MQTT_CONNECTED;
            break;
        case MQTT_EVENT_BEFORE_CONNECT:
            self->state = NATIVE_MQTT_CONNECTING;
            break;
        case MQTT_EVENT_DISCONNECTED:
        case MQTT_EVENT_ERROR:
            if (self->state != NATIVE_MQTT_IDLE) {
                self->state = NATIVE_MQTT_FAILED;
            }
            break;
        case MQTT_EVENT_SUBSCRIBED:
            self->subscribed_msg_id = event->msg_id;
            break;
        case MQTT_EVENT_DATA:
            if (
                event->current_data_offset != 0 ||
                event->data_len != event->total_data_len ||
                event->topic_len <= 0 ||
                event->topic_len >= NATIVE_MQTT_TOPIC_SIZE ||
                event->data_len < 0 ||
                event->data_len > NATIVE_MQTT_PAYLOAD_SIZE
            ) {
                break;
            }
            portENTER_CRITICAL(&self->message_lock);
            if (!self->message_pending) {
                memcpy(self->topic, event->topic, event->topic_len);
                self->topic[event->topic_len] = '\0';
                memcpy(self->payload, event->data, event->data_len);
                self->topic_len = event->topic_len;
                self->payload_len = event->data_len;
                self->message_pending = true;
            }
            portEXIT_CRITICAL(&self->message_lock);
            break;
        default:
            break;
    }
}

static void native_mqtt_check_result(int result) {
    if (result < 0) {
        mp_raise_OSError(MP_EIO);
    }
}

static mp_obj_t native_mqtt_client_make_new(
    const mp_obj_type_t *type,
    size_t n_args,
    size_t n_kw,
    const mp_obj_t *all_args
) {
    enum {
        ARG_client_id,
        ARG_server,
        ARG_port,
        ARG_user,
        ARG_password,
        ARG_keepalive,
        ARG_last_will_topic,
        ARG_last_will_msg,
        ARG_last_will_retain,
    };
    static const mp_arg_t allowed_args[] = {
        { MP_QSTR_client_id, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_obj = MP_OBJ_NULL} },
        { MP_QSTR_server, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_obj = MP_OBJ_NULL} },
        { MP_QSTR_port, MP_ARG_INT, {.u_int = 1883} },
        { MP_QSTR_user, MP_ARG_OBJ, {.u_obj = mp_const_none} },
        { MP_QSTR_password, MP_ARG_OBJ, {.u_obj = mp_const_none} },
        { MP_QSTR_keepalive, MP_ARG_INT, {.u_int = 0} },
        { MP_QSTR_last_will_topic, MP_ARG_OBJ, {.u_obj = mp_const_none} },
        { MP_QSTR_last_will_msg, MP_ARG_OBJ, {.u_obj = mp_const_none} },
        { MP_QSTR_last_will_retain, MP_ARG_BOOL, {.u_bool = true} },
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

    const char *client_id = mp_obj_str_get_str(args[ARG_client_id].u_obj);
    const char *server = mp_obj_str_get_str(args[ARG_server].u_obj);
    const char *user = args[ARG_user].u_obj == mp_const_none
        ? NULL
        : mp_obj_str_get_str(args[ARG_user].u_obj);
    const char *password = args[ARG_password].u_obj == mp_const_none
        ? NULL
        : mp_obj_str_get_str(args[ARG_password].u_obj);
    const char *last_will_topic = args[ARG_last_will_topic].u_obj == mp_const_none
        ? NULL
        : mp_obj_str_get_str(args[ARG_last_will_topic].u_obj);
    size_t last_will_len = 0;
    const char *last_will_msg = args[ARG_last_will_msg].u_obj == mp_const_none
        ? NULL
        : mp_obj_str_get_data(args[ARG_last_will_msg].u_obj, &last_will_len);
    int keepalive = args[ARG_keepalive].u_int;

    native_mqtt_client_obj_t *self = mp_obj_malloc_with_finaliser(
        native_mqtt_client_obj_t,
        type
    );
    self->client = NULL;
    self->callback = mp_const_none;
    self->state = NATIVE_MQTT_IDLE;
    self->subscribed_msg_id = 0;
    self->started = false;
    self->message_pending = false;
    self->message_lock = (portMUX_TYPE)portMUX_INITIALIZER_UNLOCKED;
    self->topic_len = 0;
    self->payload_len = 0;

    esp_mqtt_client_config_t config = {
        .broker.address.hostname = server,
        .broker.address.port = args[ARG_port].u_int,
        .broker.address.transport = MQTT_TRANSPORT_OVER_TCP,
        .credentials.client_id = client_id,
        .credentials.username = user,
        .credentials.authentication.password = password,
        .session.keepalive = keepalive,
        .session.disable_keepalive = keepalive == 0,
        .session.protocol_ver = MQTT_PROTOCOL_V_3_1_1,
        .session.last_will.topic = last_will_topic,
        .session.last_will.msg = last_will_msg,
        .session.last_will.msg_len = last_will_len,
        .session.last_will.qos = 0,
        .session.last_will.retain = args[ARG_last_will_retain].u_bool,
        .task.stack_size = NATIVE_MQTT_TASK_STACK_SIZE,
        .buffer.size = NATIVE_MQTT_BUFFER_SIZE,
        .buffer.out_size = NATIVE_MQTT_BUFFER_SIZE,
    };
    self->client = esp_mqtt_client_init(&config);
    if (self->client == NULL) {
        mp_raise_OSError(MP_ENOMEM);
    }
    esp_err_t err = esp_mqtt_client_register_event(
        self->client,
        ESP_EVENT_ANY_ID,
        native_mqtt_event_handler,
        self
    );
    if (err != ESP_OK) {
        esp_mqtt_client_destroy(self->client);
        self->client = NULL;
        mp_raise_OSError(MP_EIO);
    }
    return MP_OBJ_FROM_PTR(self);
}

static mp_obj_t native_mqtt_connect(mp_obj_t self_in) {
    native_mqtt_client_obj_t *self = MP_OBJ_TO_PTR(self_in);
    if (self->client == NULL || self->started) {
        mp_raise_OSError(MP_EINVAL);
    }
    self->state = NATIVE_MQTT_CONNECTING;
    if (esp_mqtt_client_start(self->client) != ESP_OK) {
        self->state = NATIVE_MQTT_FAILED;
        mp_raise_OSError(MP_EIO);
    }
    self->started = true;

    return MP_OBJ_NEW_SMALL_INT(0);
}
static MP_DEFINE_CONST_FUN_OBJ_1(native_mqtt_connect_obj, native_mqtt_connect);

static mp_obj_t native_mqtt_status(mp_obj_t self_in) {
    native_mqtt_client_obj_t *self = MP_OBJ_TO_PTR(self_in);
    return MP_OBJ_NEW_SMALL_INT((int)self->state);
}
static MP_DEFINE_CONST_FUN_OBJ_1(native_mqtt_status_obj, native_mqtt_status);

static mp_obj_t native_mqtt_set_callback(mp_obj_t self_in, mp_obj_t callback) {
    native_mqtt_client_obj_t *self = MP_OBJ_TO_PTR(self_in);
    if (callback != mp_const_none && !mp_obj_is_callable(callback)) {
        mp_raise_TypeError(MP_ERROR_TEXT("callback must be callable"));
    }
    self->callback = callback;
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(native_mqtt_set_callback_obj, native_mqtt_set_callback);

static mp_obj_t native_mqtt_publish(
    size_t n_pos_args,
    const mp_obj_t *pos_args,
    mp_map_t *kw_args
) {
    enum {
        ARG_topic,
        ARG_msg,
        ARG_retain,
        ARG_qos,
    };
    static const mp_arg_t allowed_args[] = {
        { MP_QSTR_topic, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_obj = MP_OBJ_NULL} },
        { MP_QSTR_msg, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_obj = MP_OBJ_NULL} },
        { MP_QSTR_retain, MP_ARG_BOOL, {.u_bool = false} },
        { MP_QSTR_qos, MP_ARG_INT, {.u_int = 0} },
    };
    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(
        n_pos_args - 1,
        pos_args + 1,
        kw_args,
        MP_ARRAY_SIZE(allowed_args),
        allowed_args,
        args
    );
    native_mqtt_client_obj_t *self = MP_OBJ_TO_PTR(pos_args[0]);
    if (self->client == NULL || self->state != NATIVE_MQTT_CONNECTED) {
        mp_raise_OSError(MP_ENOTCONN);
    }
    const char *topic = mp_obj_str_get_str(args[ARG_topic].u_obj);
    size_t payload_len = 0;
    const char *payload = mp_obj_str_get_data(args[ARG_msg].u_obj, &payload_len);
    if (args[ARG_qos].u_int != 0) {
        mp_raise_ValueError(MP_ERROR_TEXT("native MQTT supports QoS 0 only"));
    }
    int result = esp_mqtt_client_enqueue(
        self->client,
        topic,
        payload,
        payload_len,
        0,
        args[ARG_retain].u_bool,
        // ESP-IDF only queues QoS 0 messages when outbox storage is explicit.
        true
    );
    native_mqtt_check_result(result);
    return MP_OBJ_NEW_SMALL_INT(result);
}
static MP_DEFINE_CONST_FUN_OBJ_KW(native_mqtt_publish_obj, 3, native_mqtt_publish);

static mp_obj_t native_mqtt_subscribe(size_t n_args, const mp_obj_t *args) {
    native_mqtt_client_obj_t *self = MP_OBJ_TO_PTR(args[0]);
    if (self->client == NULL || self->state != NATIVE_MQTT_CONNECTED) {
        mp_raise_OSError(MP_ENOTCONN);
    }
    const char *topic = mp_obj_str_get_str(args[1]);
    int qos = n_args >= 3 ? mp_obj_get_int(args[2]) : 0;
    self->subscribed_msg_id = 0;
    int result = esp_mqtt_client_subscribe_single(self->client, topic, qos);
    native_mqtt_check_result(result);

    uint32_t start_ms = mp_hal_ticks_ms();
    while (self->subscribed_msg_id != result) {
        if (self->state != NATIVE_MQTT_CONNECTED) {
            mp_raise_OSError(MP_ENOTCONN);
        }
        if ((uint32_t)(mp_hal_ticks_ms() - start_ms) >= NATIVE_MQTT_SUBSCRIBE_TIMEOUT_MS) {
            mp_raise_OSError(MP_ETIMEDOUT);
        }
        mp_hal_delay_ms(10);
    }
    return MP_OBJ_NEW_SMALL_INT(result);
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(native_mqtt_subscribe_obj, 2, 3, native_mqtt_subscribe);

static mp_obj_t native_mqtt_check_msg(mp_obj_t self_in) {
    native_mqtt_client_obj_t *self = MP_OBJ_TO_PTR(self_in);
    if (self->state == NATIVE_MQTT_FAILED) {
        mp_raise_OSError(MP_ENOTCONN);
    }
    if (!self->message_pending || self->callback == mp_const_none) {
        return mp_const_none;
    }

    char topic[NATIVE_MQTT_TOPIC_SIZE];
    uint8_t payload[NATIVE_MQTT_PAYLOAD_SIZE];
    size_t topic_len;
    size_t payload_len;
    portENTER_CRITICAL(&self->message_lock);
    topic_len = self->topic_len;
    payload_len = self->payload_len;
    memcpy(topic, self->topic, topic_len);
    memcpy(payload, self->payload, payload_len);
    self->message_pending = false;
    portEXIT_CRITICAL(&self->message_lock);

    mp_obj_t topic_obj = mp_obj_new_bytes((const byte *)topic, topic_len);
    mp_obj_t payload_obj = mp_obj_new_bytes(payload, payload_len);
    mp_call_function_2(self->callback, topic_obj, payload_obj);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(native_mqtt_check_msg_obj, native_mqtt_check_msg);

static mp_obj_t native_mqtt_disconnect(mp_obj_t self_in) {
    native_mqtt_client_obj_t *self = MP_OBJ_TO_PTR(self_in);
    self->state = NATIVE_MQTT_IDLE;
    if (self->client != NULL && self->started) {
        esp_mqtt_client_stop(self->client);
    }
    self->started = false;
    self->message_pending = false;
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(native_mqtt_disconnect_obj, native_mqtt_disconnect);

static mp_obj_t native_mqtt_deinit(mp_obj_t self_in) {
    native_mqtt_client_obj_t *self = MP_OBJ_TO_PTR(self_in);
    native_mqtt_disconnect(self_in);
    if (self->client != NULL) {
        esp_mqtt_client_destroy(self->client);
        self->client = NULL;
    }
    self->callback = mp_const_none;
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(native_mqtt_deinit_obj, native_mqtt_deinit);

static const mp_rom_map_elem_t native_mqtt_client_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR_connect), MP_ROM_PTR(&native_mqtt_connect_obj) },
    { MP_ROM_QSTR(MP_QSTR_status), MP_ROM_PTR(&native_mqtt_status_obj) },
    { MP_ROM_QSTR(MP_QSTR_set_callback), MP_ROM_PTR(&native_mqtt_set_callback_obj) },
    { MP_ROM_QSTR(MP_QSTR_publish), MP_ROM_PTR(&native_mqtt_publish_obj) },
    { MP_ROM_QSTR(MP_QSTR_subscribe), MP_ROM_PTR(&native_mqtt_subscribe_obj) },
    { MP_ROM_QSTR(MP_QSTR_check_msg), MP_ROM_PTR(&native_mqtt_check_msg_obj) },
    { MP_ROM_QSTR(MP_QSTR_disconnect), MP_ROM_PTR(&native_mqtt_disconnect_obj) },
    { MP_ROM_QSTR(MP_QSTR_deinit), MP_ROM_PTR(&native_mqtt_deinit_obj) },
    { MP_ROM_QSTR(MP_QSTR___del__), MP_ROM_PTR(&native_mqtt_deinit_obj) },
};
static MP_DEFINE_CONST_DICT(native_mqtt_client_locals_dict, native_mqtt_client_locals_dict_table);

MP_DEFINE_CONST_OBJ_TYPE(
    native_mqtt_client_type,
    MP_QSTR_MQTTClient,
    MP_TYPE_FLAG_NONE,
    make_new, native_mqtt_client_make_new,
    locals_dict, &native_mqtt_client_locals_dict
);

static const mp_rom_map_elem_t native_mqtt_module_globals_table[] = {
    { MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_espectre_native_mqtt) },
    { MP_ROM_QSTR(MP_QSTR_MQTTClient), MP_ROM_PTR(&native_mqtt_client_type) },
};
static MP_DEFINE_CONST_DICT(native_mqtt_module_globals, native_mqtt_module_globals_table);

const mp_obj_module_t native_mqtt_module = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t *)&native_mqtt_module_globals,
};

MP_REGISTER_MODULE(MP_QSTR_espectre_native_mqtt, native_mqtt_module);
