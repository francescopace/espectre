// SPDX-License-Identifier: GPL-3.0-only
// Commercial licensing available under separate agreement; see LICENSING.md.

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cJSON.h"
#include "esp_err.h"
#include "esp_http_server.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "mdns.h"
#include "py/mperrno.h"
#include "py/runtime.h"

#define DIRECT_REQUEST_PATH "/espectre/v1/request"
#define DIRECT_EVENTS_PATH "/espectre/v1/events"
#define DIRECT_MAX_REQUEST_BYTES (512)
#define DIRECT_MAX_EVENT_BYTES (4096)
#define DIRECT_MAX_STATUS_BYTES (384)
#define DIRECT_MAX_PROTOCOL_VERSION_BYTES (16)
#define DIRECT_MAX_DNS_SD_SCHEMA_VERSION_BYTES (8)
#define DIRECT_MAX_COMMAND_ID_BYTES (64)
#define DIRECT_MAX_COMMAND_BYTES (48)

typedef struct {
    httpd_handle_t server;
    httpd_req_t *event_request;
    SemaphoreHandle_t lock;
    char *capabilities;
    char *info;
    char status[DIRECT_MAX_STATUS_BYTES + 1];
    char *config;
    char *device_id;
    char *protocol_version;
    char *dns_sd_schema_version;
    bool mdns_service_added;
} native_direct_state_t;

static native_direct_state_t direct_state;

static bool direct_identifier_valid(const char *value, size_t max_length, bool command) {
    if (value == NULL) {
        return false;
    }
    size_t length = strlen(value);
    if (length == 0 || length > max_length) {
        return false;
    }
    for (size_t index = 0; index < length; ++index) {
        char character = value[index];
        bool alphanumeric =
            (character >= '0' && character <= '9') ||
            (character >= 'A' && character <= 'Z') ||
            (character >= 'a' && character <= 'z');
        if (
            !alphanumeric && character != '_' && character != '-' && character != '.' &&
            (command || character != ':')
        ) {
            return false;
        }
    }
    return true;
}

static bool direct_origin_allowed(const char *origin) {
    return origin == NULL || origin[0] == '\0' ||
        strcmp(origin, "https://espectre.dev") == 0 ||
        strcmp(origin, "https://www.espectre.dev") == 0 ||
        strcmp(origin, "https://test.espectre.dev") == 0 ||
        strcmp(origin, "http://localhost") == 0 ||
        strcmp(origin, "http://127.0.0.1") == 0;
}

static bool direct_read_origin(httpd_req_t *request, char *origin, size_t capacity) {
    size_t length = httpd_req_get_hdr_value_len(request, "Origin");
    if (length == 0) {
        origin[0] = '\0';
        return true;
    }
    if (length >= capacity || httpd_req_get_hdr_value_str(request, "Origin", origin, capacity) != ESP_OK) {
        return false;
    }
    return direct_origin_allowed(origin);
}

static bool direct_set_cors(httpd_req_t *request) {
    char origin[96];
    if (!direct_read_origin(request, origin, sizeof(origin))) {
        return false;
    }
    if (origin[0] != '\0') {
        (void)httpd_resp_set_hdr(request, "Access-Control-Allow-Origin", origin);
        (void)httpd_resp_set_hdr(request, "Vary", "Origin");
    }
    return true;
}

static char *direct_replace_string(char **target, const char *source, size_t length) {
    char *replacement = malloc(length + 1);
    if (replacement == NULL) {
        return NULL;
    }
    memcpy(replacement, source, length);
    replacement[length] = '\0';
    if (direct_state.lock != NULL) {
        xSemaphoreTake(direct_state.lock, portMAX_DELAY);
    }
    char *previous = *target;
    *target = replacement;
    if (direct_state.lock != NULL) {
        xSemaphoreGive(direct_state.lock);
    }
    free(previous);
    return replacement;
}

static bool direct_replace_status(const char *source, size_t length) {
    if (length > DIRECT_MAX_STATUS_BYTES) {
        return false;
    }
    if (direct_state.lock != NULL) {
        xSemaphoreTake(direct_state.lock, portMAX_DELAY);
    }
    memcpy(direct_state.status, source, length);
    direct_state.status[length] = '\0';
    if (direct_state.lock != NULL) {
        xSemaphoreGive(direct_state.lock);
    }
    return true;
}

static esp_err_t direct_send_json(
    httpd_req_t *request,
    const char *status,
    const char *body
) {
    if (!direct_set_cors(request)) {
        httpd_resp_set_status(request, "403 Forbidden");
        return httpd_resp_sendstr(request, "Forbidden");
    }
    httpd_resp_set_status(request, status);
    httpd_resp_set_type(request, "application/json");
    (void)httpd_resp_set_hdr(request, "Cache-Control", "no-store");
    return httpd_resp_sendstr(request, body);
}

static esp_err_t direct_send_result(
    httpd_req_t *request,
    const char *command_id,
    const char *command,
    bool accepted,
    const char *code,
    const char *message,
    const char *snapshot,
    const char *http_status
) {
    cJSON *root = cJSON_CreateObject();
    if (root == NULL) {
        return ESP_ERR_NO_MEM;
    }
    cJSON_AddStringToObject(
        root,
        "protocol_version",
        direct_state.protocol_version == NULL ? "" : direct_state.protocol_version
    );
    cJSON_AddStringToObject(root, "device_id", direct_state.device_id == NULL ? "" : direct_state.device_id);
    cJSON_AddStringToObject(root, "command_id", command_id == NULL ? "" : command_id);
    cJSON_AddStringToObject(root, "command", command == NULL ? "" : command);
    cJSON_AddBoolToObject(root, "accepted", accepted);
    cJSON_AddStringToObject(root, "code", code);
    cJSON_AddStringToObject(root, "message", message);
    if (snapshot != NULL) {
        cJSON *data = cJSON_Parse(snapshot);
        if (data == NULL) {
            cJSON_Delete(root);
            return ESP_ERR_INVALID_STATE;
        }
        cJSON_AddItemToObject(root, "data", data);
    }
    char *body = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    if (body == NULL) {
        return ESP_ERR_NO_MEM;
    }
    esp_err_t result = direct_send_json(request, http_status, body);
    cJSON_free(body);
    return result;
}

static char *direct_copy_command_snapshot(const char *command) {
    if (direct_state.lock != NULL) {
        xSemaphoreTake(direct_state.lock, portMAX_DELAY);
    }
    const char *snapshot = NULL;
    if (strcmp(command, "capabilities") == 0) {
        snapshot = direct_state.capabilities;
    } else if (strcmp(command, "info") == 0) {
        snapshot = direct_state.info;
    } else if (strcmp(command, "status") == 0) {
        snapshot = direct_state.status;
    } else if (strcmp(command, "config") == 0) {
        snapshot = direct_state.config;
    }
    char *copy = snapshot == NULL ? NULL : strdup(snapshot);
    if (direct_state.lock != NULL) {
        xSemaphoreGive(direct_state.lock);
    }
    return copy;
}

static esp_err_t direct_request_handler(httpd_req_t *request) {
    if (!direct_set_cors(request)) {
        httpd_resp_set_status(request, "403 Forbidden");
        return httpd_resp_sendstr(request, "Forbidden");
    }
    if (request->content_len <= 0 || request->content_len > DIRECT_MAX_REQUEST_BYTES) {
        return direct_send_result(
            request, "", "", false, "invalid_params", "request body is invalid", NULL,
            "400 Bad Request"
        );
    }
    char content_type[48];
    if (
        httpd_req_get_hdr_value_str(request, "Content-Type", content_type, sizeof(content_type)) != ESP_OK ||
        strcmp(content_type, "application/json") != 0
    ) {
        return direct_send_result(
            request, "", "", false, "invalid_params", "content type must be application/json", NULL,
            "415 Unsupported Media Type"
        );
    }
    char payload[DIRECT_MAX_REQUEST_BYTES + 1];
    size_t received_total = 0;
    while (received_total < (size_t)request->content_len) {
        int received = httpd_req_recv(
            request,
            payload + received_total,
            (size_t)request->content_len - received_total
        );
        if (received <= 0) {
            return ESP_FAIL;
        }
        received_total += (size_t)received;
    }
    payload[received_total] = '\0';

    cJSON *root = cJSON_ParseWithLength(payload, received_total);
    const cJSON *version = root == NULL ? NULL : cJSON_GetObjectItemCaseSensitive(root, "protocol_version");
    const cJSON *command_id = root == NULL ? NULL : cJSON_GetObjectItemCaseSensitive(root, "command_id");
    const cJSON *command = root == NULL ? NULL : cJSON_GetObjectItemCaseSensitive(root, "command");
    bool identifiers_valid = cJSON_IsObject(root) && cJSON_IsString(command_id) &&
        cJSON_IsString(command) &&
        direct_identifier_valid(command_id->valuestring, DIRECT_MAX_COMMAND_ID_BYTES, false) &&
        direct_identifier_valid(command->valuestring, DIRECT_MAX_COMMAND_BYTES, true);
    if (!identifiers_valid) {
        cJSON_Delete(root);
        return direct_send_result(
            request, "", "", false, "invalid_params", "request message is invalid", NULL,
            "400 Bad Request"
        );
    }
    if (
        !cJSON_IsString(version) || direct_state.protocol_version == NULL ||
        strcmp(version->valuestring, direct_state.protocol_version) != 0
    ) {
        esp_err_t result = direct_send_result(
            request, command_id->valuestring, command->valuestring, false, "unsupported_version",
            "protocol_version is unsupported", NULL, "400 Bad Request"
        );
        cJSON_Delete(root);
        return result;
    }
    if (cJSON_GetArraySize(root) != 3) {
        esp_err_t result = direct_send_result(
            request, command_id->valuestring, command->valuestring, false, "invalid_params",
            "command does not accept parameters", NULL, "400 Bad Request"
        );
        cJSON_Delete(root);
        return result;
    }

    bool supported = strcmp(command->valuestring, "capabilities") == 0 ||
        strcmp(command->valuestring, "info") == 0 ||
        strcmp(command->valuestring, "status") == 0 ||
        strcmp(command->valuestring, "config") == 0;
    char *snapshot = supported ? direct_copy_command_snapshot(command->valuestring) : NULL;
    esp_err_t result;
    if (!supported) {
        result = direct_send_result(
            request, command_id->valuestring, command->valuestring, false, "unsupported",
            "command is not supported", NULL, "200 OK"
        );
    } else if (snapshot == NULL) {
        result = direct_send_result(
            request, command_id->valuestring, command->valuestring, false, "unavailable",
            "snapshot is unavailable", NULL, "503 Service Unavailable"
        );
    } else {
        result = direct_send_result(
            request, command_id->valuestring, command->valuestring, true, "ok",
            "query completed", snapshot, "200 OK"
        );
    }
    free(snapshot);
    cJSON_Delete(root);
    return result;
}

static esp_err_t direct_options_handler(httpd_req_t *request) {
    if (!direct_set_cors(request)) {
        httpd_resp_set_status(request, "403 Forbidden");
        return httpd_resp_sendstr(request, "Forbidden");
    }
    (void)httpd_resp_set_hdr(request, "Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    (void)httpd_resp_set_hdr(request, "Access-Control-Allow-Headers", "Content-Type");
    char private_network[8];
    if (
        httpd_req_get_hdr_value_str(
            request,
            "Access-Control-Request-Private-Network",
            private_network,
            sizeof(private_network)
        ) == ESP_OK && strcmp(private_network, "true") == 0
    ) {
        (void)httpd_resp_set_hdr(request, "Access-Control-Allow-Private-Network", "true");
    }
    httpd_resp_set_status(request, "204 No Content");
    return httpd_resp_send(request, NULL, 0);
}

static esp_err_t direct_events_handler(httpd_req_t *request) {
    if (!direct_set_cors(request)) {
        httpd_resp_set_status(request, "403 Forbidden");
        return httpd_resp_sendstr(request, "Forbidden");
    }
    if (direct_state.lock != NULL) {
        xSemaphoreTake(direct_state.lock, portMAX_DELAY);
    }
    bool occupied = direct_state.event_request != NULL;
    if (direct_state.lock != NULL) {
        xSemaphoreGive(direct_state.lock);
    }
    if (occupied) {
        httpd_resp_set_status(request, "503 Service Unavailable");
        return httpd_resp_sendstr(request, "event stream is already in use");
    }

    httpd_req_t *async_request = NULL;
    if (httpd_req_async_handler_begin(request, &async_request) != ESP_OK || async_request == NULL) {
        return ESP_FAIL;
    }
    httpd_resp_set_type(async_request, "text/event-stream");
    (void)httpd_resp_set_hdr(async_request, "Cache-Control", "no-cache");
    (void)httpd_resp_set_hdr(async_request, "Connection", "keep-alive");
    if (!direct_set_cors(async_request) ||
        httpd_resp_send_chunk(async_request, ": connected\n\n", strlen(": connected\n\n")) != ESP_OK) {
        httpd_req_async_handler_complete(async_request);
        return ESP_FAIL;
    }
    if (direct_state.lock != NULL) {
        xSemaphoreTake(direct_state.lock, portMAX_DELAY);
    }
    direct_state.event_request = async_request;
    if (direct_state.lock != NULL) {
        xSemaphoreGive(direct_state.lock);
    }
    return ESP_OK;
}

static void direct_close_event_stream(void) {
    if (direct_state.lock != NULL) {
        xSemaphoreTake(direct_state.lock, portMAX_DELAY);
    }
    httpd_req_t *request = direct_state.event_request;
    direct_state.event_request = NULL;
    if (direct_state.lock != NULL) {
        xSemaphoreGive(direct_state.lock);
    }
    if (request != NULL) {
        (void)httpd_resp_send_chunk(request, NULL, 0);
        httpd_req_async_handler_complete(request);
    }
}

static void direct_free_snapshots(void) {
    free(direct_state.capabilities);
    free(direct_state.info);
    free(direct_state.config);
    free(direct_state.device_id);
    free(direct_state.protocol_version);
    free(direct_state.dns_sd_schema_version);
    direct_state.capabilities = NULL;
    direct_state.info = NULL;
    direct_state.status[0] = '\0';
    direct_state.config = NULL;
    direct_state.device_id = NULL;
    direct_state.protocol_version = NULL;
    direct_state.dns_sd_schema_version = NULL;
}

static void direct_stop_native(void) {
    direct_close_event_stream();
    if (direct_state.server != NULL) {
        httpd_stop(direct_state.server);
        direct_state.server = NULL;
    }
    if (direct_state.mdns_service_added) {
        (void)mdns_service_remove("_espectre", "_tcp");
        direct_state.mdns_service_added = false;
    }
    direct_free_snapshots();
}

static bool direct_register_http_handlers(void) {
    const httpd_uri_t request_handler = {
        .uri = DIRECT_REQUEST_PATH,
        .method = HTTP_POST,
        .handler = direct_request_handler,
    };
    const httpd_uri_t options_handler = {
        .uri = DIRECT_REQUEST_PATH,
        .method = HTTP_OPTIONS,
        .handler = direct_options_handler,
    };
    const httpd_uri_t events_handler = {
        .uri = DIRECT_EVENTS_PATH,
        .method = HTTP_GET,
        .handler = direct_events_handler,
    };
    const httpd_uri_t events_options_handler = {
        .uri = DIRECT_EVENTS_PATH,
        .method = HTTP_OPTIONS,
        .handler = direct_options_handler,
    };
    return httpd_register_uri_handler(direct_state.server, &request_handler) == ESP_OK &&
        httpd_register_uri_handler(direct_state.server, &options_handler) == ESP_OK &&
        httpd_register_uri_handler(direct_state.server, &events_handler) == ESP_OK &&
        httpd_register_uri_handler(direct_state.server, &events_options_handler) == ESP_OK;
}

static bool direct_add_mdns_service(
    const char *hostname,
    const char *instance,
    const char *device_id,
    const char *chip,
    uint16_t port
) {
    esp_err_t init_result = mdns_init();
    if (init_result != ESP_OK && init_result != ESP_ERR_INVALID_STATE) {
        return false;
    }
    if (mdns_hostname_set(hostname) != ESP_OK || mdns_instance_name_set(instance) != ESP_OK) {
        return false;
    }
    mdns_txt_item_t txt[] = {
        {"txtvers", direct_state.dns_sd_schema_version},
        {"protovers", direct_state.protocol_version},
        {"device_id", device_id},
        {"name", instance},
        {"frontend", "micro"},
        {"transport", "http"},
        {"path", DIRECT_REQUEST_PATH},
        {"events", DIRECT_EVENTS_PATH},
        {"firmware", "micropython"},
        {"chip", chip},
        {"capabilities", "monitor"},
    };
    if (mdns_service_add(instance, "_espectre", "_tcp", port, txt, MP_ARRAY_SIZE(txt)) != ESP_OK) {
        return false;
    }
    direct_state.mdns_service_added = true;
    return true;
}

static mp_obj_t native_direct_start(
    size_t n_pos_args,
    const mp_obj_t *pos_args,
    mp_map_t *kw_args
) {
    enum {
        ARG_port,
        ARG_hostname,
        ARG_instance,
        ARG_device_id,
        ARG_chip,
        ARG_protocol_version,
        ARG_dns_sd_schema_version,
        ARG_capabilities,
        ARG_info,
        ARG_config,
        ARG_status,
    };
    static const mp_arg_t allowed_args[] = {
        {MP_QSTR_port, MP_ARG_REQUIRED | MP_ARG_INT, {.u_int = 0}},
        {MP_QSTR_hostname, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_obj = MP_OBJ_NULL}},
        {MP_QSTR_instance, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_obj = MP_OBJ_NULL}},
        {MP_QSTR_device_id, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_obj = MP_OBJ_NULL}},
        {MP_QSTR_chip, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_obj = MP_OBJ_NULL}},
        {MP_QSTR_protocol_version, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_obj = MP_OBJ_NULL}},
        {MP_QSTR_dns_sd_schema_version, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_obj = MP_OBJ_NULL}},
        {MP_QSTR_capabilities, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_obj = MP_OBJ_NULL}},
        {MP_QSTR_info, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_obj = MP_OBJ_NULL}},
        {MP_QSTR_config, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_obj = MP_OBJ_NULL}},
        {MP_QSTR_status, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_obj = MP_OBJ_NULL}},
    };
    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(n_pos_args, pos_args, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);
    if (direct_state.server != NULL) {
        mp_raise_OSError(MP_EBUSY);
    }
    int port = args[ARG_port].u_int;
    if (port <= 0 || port > 65535) {
        mp_raise_ValueError(MP_ERROR_TEXT("invalid Direct HTTP port"));
    }
    const char *hostname = mp_obj_str_get_str(args[ARG_hostname].u_obj);
    const char *instance = mp_obj_str_get_str(args[ARG_instance].u_obj);
    const char *device_id = mp_obj_str_get_str(args[ARG_device_id].u_obj);
    const char *chip = mp_obj_str_get_str(args[ARG_chip].u_obj);
    const char *protocol_version = mp_obj_str_get_str(args[ARG_protocol_version].u_obj);
    const char *dns_sd_schema_version = mp_obj_str_get_str(args[ARG_dns_sd_schema_version].u_obj);
    size_t capabilities_len;
    size_t info_len;
    size_t config_len;
    size_t status_len;
    const char *capabilities = mp_obj_str_get_data(args[ARG_capabilities].u_obj, &capabilities_len);
    const char *info = mp_obj_str_get_data(args[ARG_info].u_obj, &info_len);
    const char *config_snapshot = mp_obj_str_get_data(args[ARG_config].u_obj, &config_len);
    const char *status = mp_obj_str_get_data(args[ARG_status].u_obj, &status_len);

    size_t protocol_version_len = strlen(protocol_version);
    size_t dns_sd_schema_version_len = strlen(dns_sd_schema_version);
    if (
        protocol_version_len == 0 || protocol_version_len > DIRECT_MAX_PROTOCOL_VERSION_BYTES ||
        dns_sd_schema_version_len == 0 ||
        dns_sd_schema_version_len > DIRECT_MAX_DNS_SD_SCHEMA_VERSION_BYTES
    ) {
        mp_raise_ValueError(MP_ERROR_TEXT("invalid protocol version"));
    }

    if (direct_state.lock == NULL) {
        direct_state.lock = xSemaphoreCreateMutex();
    }
    if (direct_state.lock == NULL ||
        direct_replace_string(&direct_state.capabilities, capabilities, capabilities_len) == NULL ||
        direct_replace_string(&direct_state.info, info, info_len) == NULL ||
        direct_replace_string(&direct_state.config, config_snapshot, config_len) == NULL ||
        !direct_replace_status(status, status_len) ||
        direct_replace_string(&direct_state.device_id, device_id, strlen(device_id)) == NULL ||
        direct_replace_string(
            &direct_state.protocol_version,
            protocol_version,
            protocol_version_len
        ) == NULL ||
        direct_replace_string(
            &direct_state.dns_sd_schema_version,
            dns_sd_schema_version,
            dns_sd_schema_version_len
        ) == NULL) {
        direct_free_snapshots();
        mp_raise_OSError(MP_ENOMEM);
    }

    httpd_config_t server_config = HTTPD_DEFAULT_CONFIG();
    server_config.server_port = (uint16_t)port;
    server_config.stack_size = 3584;
    server_config.max_open_sockets = 3;
    server_config.lru_purge_enable = true;
    server_config.recv_wait_timeout = 5;
    server_config.send_wait_timeout = 5;
    if (httpd_start(&direct_state.server, &server_config) != ESP_OK || !direct_register_http_handlers()) {
        direct_stop_native();
        mp_raise_OSError(MP_EIO);
    }
    if (!direct_add_mdns_service(hostname, instance, device_id, chip, (uint16_t)port)) {
        direct_stop_native();
        mp_raise_OSError(MP_EIO);
    }
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_KW(native_direct_start_obj, 0, native_direct_start);

static mp_obj_t native_direct_update_status(mp_obj_t status_obj) {
    size_t length;
    const char *status = mp_obj_str_get_data(status_obj, &length);
    if (!direct_replace_status(status, length)) {
        mp_raise_ValueError(MP_ERROR_TEXT("Direct status is too large"));
    }
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(native_direct_update_status_obj, native_direct_update_status);

static mp_obj_t native_direct_publish(mp_obj_t event_obj, mp_obj_t payload_obj) {
    size_t event_length;
    size_t payload_length;
    const char *event = mp_obj_str_get_data(event_obj, &event_length);
    const char *payload = mp_obj_str_get_data(payload_obj, &payload_length);
    if (event_length == 0 || event_length > 32 || payload_length > DIRECT_MAX_EVENT_BYTES) {
        mp_raise_ValueError(MP_ERROR_TEXT("Direct event is too large"));
    }
    for (size_t index = 0; index < event_length; ++index) {
        char value = event[index];
        if (!((value >= 'a' && value <= 'z') || value == '_')) {
            mp_raise_ValueError(MP_ERROR_TEXT("invalid Direct event name"));
        }
    }
    if (direct_state.lock != NULL) {
        xSemaphoreTake(direct_state.lock, portMAX_DELAY);
    }
    httpd_req_t *request = direct_state.event_request;
    if (direct_state.lock != NULL) {
        xSemaphoreGive(direct_state.lock);
    }
    if (request == NULL) {
        return mp_const_false;
    }
    char prefix[80];
    int prefix_length = snprintf(
        prefix,
        sizeof(prefix),
        "event: %.*s\ndata: ",
        (int)event_length,
        event
    );
    esp_err_t result = prefix_length <= 0 || (size_t)prefix_length >= sizeof(prefix)
        ? ESP_FAIL
        : httpd_resp_send_chunk(request, prefix, (size_t)prefix_length);
    if (result == ESP_OK) {
        result = httpd_resp_send_chunk(request, payload, payload_length);
    }
    if (result == ESP_OK) {
        result = httpd_resp_send_chunk(request, "\n\n", 2);
    }
    if (result != ESP_OK) {
        direct_close_event_stream();
        return mp_const_false;
    }
    return mp_const_true;
}
static MP_DEFINE_CONST_FUN_OBJ_2(native_direct_publish_obj, native_direct_publish);

static mp_obj_t native_direct_stop(void) {
    direct_stop_native();
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_0(native_direct_stop_obj, native_direct_stop);

static const mp_rom_map_elem_t native_direct_module_globals_table[] = {
    {MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_espectre_native_direct)},
    {MP_ROM_QSTR(MP_QSTR_start), MP_ROM_PTR(&native_direct_start_obj)},
    {MP_ROM_QSTR(MP_QSTR_update_status), MP_ROM_PTR(&native_direct_update_status_obj)},
    {MP_ROM_QSTR(MP_QSTR_publish), MP_ROM_PTR(&native_direct_publish_obj)},
    {MP_ROM_QSTR(MP_QSTR_stop), MP_ROM_PTR(&native_direct_stop_obj)},
};
static MP_DEFINE_CONST_DICT(native_direct_module_globals, native_direct_module_globals_table);

const mp_obj_module_t native_direct_module = {
    .base = {&mp_type_module},
    .globals = (mp_obj_dict_t *)&native_direct_module_globals,
};

MP_REGISTER_MODULE(MP_QSTR_espectre_native_direct, native_direct_module);
