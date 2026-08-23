// SPDX-License-Identifier: GPL-3.0-only
// Commercial licensing available under separate agreement; see LICENSING.md.

#include <errno.h>
#include <fcntl.h>
#include <net/if.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include "esp_netif.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "lwip/inet.h"
#include "lwip/sockets.h"
#include "lwip/tcp.h"
#include "py/mperrno.h"
#include "py/mphal.h"
#include "py/runtime.h"

#define NATIVE_TRAFFIC_TASK_STACK_SIZE (3072)
#define NATIVE_TRAFFIC_TASK_PRIORITY (5)
#define NATIVE_TRAFFIC_REOPEN_ERROR_COUNT (8)
#define NATIVE_TRAFFIC_STOP_TIMEOUT_MS (1000)
#define NATIVE_TRAFFIC_DNS_PORT (53)
#define NATIVE_TRAFFIC_DNS_QUERY_SIZE (17)
#define NATIVE_TRAFFIC_DNS_TCP_FRAME_SIZE (NATIVE_TRAFFIC_DNS_QUERY_SIZE + 2)
#define NATIVE_TRAFFIC_TCP_RECONNECT_DELAY_US (1000000LL)

typedef enum {
    NATIVE_TRAFFIC_MODE_PING,
    NATIVE_TRAFFIC_MODE_DNS,
} native_traffic_mode_t;

typedef struct __attribute__((packed)) {
    uint8_t type;
    uint8_t code;
    uint16_t checksum;
    uint16_t identifier;
    uint16_t sequence;
} native_traffic_ping_packet_t;

static const uint8_t native_traffic_dns_query_template[] = {
    0x00, 0x00, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01,
};

typedef enum {
    NATIVE_TRAFFIC_SOCKET_READY,
    NATIVE_TRAFFIC_SOCKET_PEER_CLOSED,
    NATIVE_TRAFFIC_SOCKET_ERROR,
} native_traffic_socket_drain_result_t;

typedef enum {
    NATIVE_TRAFFIC_TCP_DISCONNECTED,
    NATIVE_TRAFFIC_TCP_CONNECTING,
    NATIVE_TRAFFIC_TCP_CONNECTED,
} native_traffic_tcp_state_t;

typedef struct _native_traffic_obj_t {
    mp_obj_base_t base;
    TaskHandle_t task;
    volatile bool running;
    volatile bool paused;
    volatile bool task_exited;
    volatile bool reopen_requested;
    volatile uint32_t packet_count;
    volatile uint32_t error_count;
    uint32_t gateway_addr;
    uint32_t rate_pps;
    uint16_t identifier;
    uint16_t sequence;
    native_traffic_mode_t mode;
    int sock;
} native_traffic_obj_t;

static uint16_t native_traffic_checksum(const void *data, size_t len) {
    const uint8_t *bytes = data;
    uint32_t sum = 0;
    while (len >= 2) {
        sum += ((uint16_t)bytes[0] << 8) | bytes[1];
        bytes += 2;
        len -= 2;
    }
    if (len == 1) {
        sum += (uint16_t)bytes[0] << 8;
    }
    while (sum >> 16) {
        sum = (sum & 0xffff) + (sum >> 16);
    }
    return (uint16_t)~sum;
}

static int native_traffic_open_socket(const native_traffic_obj_t *self) {
    int socket_type = self->mode == NATIVE_TRAFFIC_MODE_PING ? SOCK_RAW : SOCK_STREAM;
    int socket_protocol = self->mode == NATIVE_TRAFFIC_MODE_PING ? IPPROTO_ICMP : IPPROTO_TCP;
    int sock = socket(AF_INET, socket_type, socket_protocol);
    if (sock < 0) {
        return -1;
    }
    int flags = fcntl(sock, F_GETFL, 0);
    if (flags < 0 || fcntl(sock, F_SETFL, flags | O_NONBLOCK) < 0) {
        close(sock);
        return -1;
    }
    // Keep sensing traffic on the station interface, matching the shared
    // ESP-IDF traffic generator when other lwIP interfaces are present.
    esp_netif_t *netif = esp_netif_get_handle_from_ifkey("WIFI_STA_DEF");
    if (netif != NULL) {
        int if_index = esp_netif_get_netif_impl_index(netif);
        struct ifreq iface = {0};
        if (
            if_index > 0 &&
            if_indextoname((unsigned int)if_index, iface.ifr_name) != NULL
        ) {
            (void)setsockopt(
                sock,
                SOL_SOCKET,
                SO_BINDTODEVICE,
                &iface,
                sizeof(iface)
            );
        }
    }
    // Match the production ESP-IDF traffic generator's low-latency WMM hint.
    int sensing_tos = 46 << 2;
    (void)setsockopt(sock, IPPROTO_IP, IP_TOS, &sensing_tos, sizeof(sensing_tos));
    if (self->mode == NATIVE_TRAFFIC_MODE_DNS) {
        int enabled = 1;
        (void)setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &enabled, sizeof(enabled));
    }
    return sock;
}

static void native_traffic_close_socket(native_traffic_obj_t *self) {
    if (self->sock >= 0) {
        close(self->sock);
        self->sock = -1;
    }
}

static native_traffic_socket_drain_result_t native_traffic_drain_socket(int sock) {
    uint8_t buffer[128];
    while (true) {
        ssize_t received = recv(sock, buffer, sizeof(buffer), MSG_DONTWAIT);
        if (received > 0) {
            continue;
        }
        if (received == 0) {
            return NATIVE_TRAFFIC_SOCKET_PEER_CLOSED;
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return NATIVE_TRAFFIC_SOCKET_READY;
        }
        return NATIVE_TRAFFIC_SOCKET_ERROR;
    }
}

static native_traffic_tcp_state_t native_traffic_start_tcp_connect(
    int sock,
    const struct sockaddr_in *destination
) {
    if (
        connect(
            sock,
            (const struct sockaddr *)destination,
            sizeof(*destination)
        ) == 0
    ) {
        return NATIVE_TRAFFIC_TCP_CONNECTED;
    }
    if (errno == EISCONN) {
        return NATIVE_TRAFFIC_TCP_CONNECTED;
    }
    if (errno == EINPROGRESS || errno == EALREADY) {
        return NATIVE_TRAFFIC_TCP_CONNECTING;
    }
    return NATIVE_TRAFFIC_TCP_DISCONNECTED;
}

static native_traffic_tcp_state_t native_traffic_poll_tcp_connect(int sock) {
    fd_set write_fds;
    fd_set error_fds;
    FD_ZERO(&write_fds);
    FD_ZERO(&error_fds);
    FD_SET(sock, &write_fds);
    FD_SET(sock, &error_fds);
    struct timeval timeout = {0};
    int ready = select(sock + 1, NULL, &write_fds, &error_fds, &timeout);
    if (ready == 0) {
        return NATIVE_TRAFFIC_TCP_CONNECTING;
    }
    if (ready < 0) {
        return NATIVE_TRAFFIC_TCP_DISCONNECTED;
    }

    int socket_error = 0;
    socklen_t error_len = sizeof(socket_error);
    if (
        getsockopt(sock, SOL_SOCKET, SO_ERROR, &socket_error, &error_len) != 0 ||
        socket_error != 0
    ) {
        if (socket_error != 0) {
            errno = socket_error;
        }
        return NATIVE_TRAFFIC_TCP_DISCONNECTED;
    }
    return NATIVE_TRAFFIC_TCP_CONNECTED;
}

static int64_t native_traffic_next_send_deadline_us(
    int64_t previous_deadline_us,
    int64_t send_started_us,
    int64_t interval_us
) {
    if (interval_us <= 0) {
        return send_started_us;
    }
    if (previous_deadline_us <= 0) {
        return send_started_us + interval_us;
    }

    int64_t phase_deadline_us = previous_deadline_us + interval_us;
    int64_t remaining_us = phase_deadline_us - send_started_us;
    if (remaining_us < interval_us / 2) {
        return send_started_us + interval_us;
    }
    return phase_deadline_us;
}

static ssize_t native_traffic_send_packet(
    native_traffic_obj_t *self,
    const struct sockaddr_in *destination
) {
    if (self->mode == NATIVE_TRAFFIC_MODE_DNS) {
        uint8_t frame[NATIVE_TRAFFIC_DNS_TCP_FRAME_SIZE];
        uint16_t transaction_id = ++self->sequence;
        frame[0] = 0;
        frame[1] = NATIVE_TRAFFIC_DNS_QUERY_SIZE;
        memcpy(
            frame + 2,
            native_traffic_dns_query_template,
            sizeof(native_traffic_dns_query_template)
        );
        frame[2] = (uint8_t)(transaction_id >> 8);
        frame[3] = (uint8_t)(transaction_id & 0xff);
        ssize_t sent = send(self->sock, frame, sizeof(frame), MSG_DONTWAIT);
        if (sent >= 0 && (size_t)sent != sizeof(frame)) {
            errno = EIO;
            return -1;
        }
        return sent;
    }

    native_traffic_ping_packet_t packet = {
        .type = 8,
        .code = 0,
        .checksum = 0,
        .identifier = htons(self->identifier),
        .sequence = htons(++self->sequence),
    };
    packet.checksum = htons(native_traffic_checksum(&packet, sizeof(packet)));
    return sendto(
        self->sock,
        &packet,
        sizeof(packet),
        0,
        (const struct sockaddr *)destination,
        sizeof(*destination)
    );
}

static void native_traffic_task(void *arg) {
    native_traffic_obj_t *self = arg;
    struct sockaddr_in destination = {
        .sin_family = AF_INET,
        .sin_port = htons(
            self->mode == NATIVE_TRAFFIC_MODE_DNS ? NATIVE_TRAFFIC_DNS_PORT : 0
        ),
        .sin_addr.s_addr = self->gateway_addr,
    };
    uint32_t consecutive_errors = 0;
    int64_t next_send_deadline_us = 0;
    int64_t next_connect_attempt_us = 0;
    native_traffic_tcp_state_t connection_state =
        self->mode == NATIVE_TRAFFIC_MODE_DNS
            ? NATIVE_TRAFFIC_TCP_DISCONNECTED
            : NATIVE_TRAFFIC_TCP_CONNECTED;

    while (self->running) {
        if (self->paused) {
            vTaskDelay(pdMS_TO_TICKS(50));
            continue;
        }
        if (self->reopen_requested || self->sock < 0) {
            self->reopen_requested = false;
            native_traffic_close_socket(self);
            self->sock = native_traffic_open_socket(self);
            if (self->sock < 0) {
                self->error_count++;
                vTaskDelay(pdMS_TO_TICKS(100));
                continue;
            }
            consecutive_errors = 0;
            next_send_deadline_us = 0;
            next_connect_attempt_us = 0;
            connection_state = self->mode == NATIVE_TRAFFIC_MODE_DNS
                ? NATIVE_TRAFFIC_TCP_DISCONNECTED
                : NATIVE_TRAFFIC_TCP_CONNECTED;
        }

        if (
            self->mode == NATIVE_TRAFFIC_MODE_DNS &&
            connection_state != NATIVE_TRAFFIC_TCP_CONNECTED
        ) {
            int64_t now_us = esp_timer_get_time();
            if (connection_state == NATIVE_TRAFFIC_TCP_DISCONNECTED) {
                if (now_us < next_connect_attempt_us) {
                    vTaskDelay(pdMS_TO_TICKS(10));
                    continue;
                }
                connection_state = native_traffic_start_tcp_connect(
                    self->sock,
                    &destination
                );
            } else {
                connection_state = native_traffic_poll_tcp_connect(self->sock);
            }
            if (connection_state == NATIVE_TRAFFIC_TCP_DISCONNECTED) {
                native_traffic_close_socket(self);
                self->sock = native_traffic_open_socket(self);
                if (self->sock < 0) {
                    self->error_count++;
                }
                next_connect_attempt_us =
                    esp_timer_get_time() + NATIVE_TRAFFIC_TCP_RECONNECT_DELAY_US;
                next_send_deadline_us = 0;
            }
            if (connection_state != NATIVE_TRAFFIC_TCP_CONNECTED) {
                vTaskDelay(pdMS_TO_TICKS(10));
                continue;
            }
            next_send_deadline_us = 0;
        }

        native_traffic_socket_drain_result_t drain_result =
            native_traffic_drain_socket(self->sock);
        if (
            self->mode == NATIVE_TRAFFIC_MODE_DNS &&
            drain_result != NATIVE_TRAFFIC_SOCKET_READY
        ) {
            native_traffic_close_socket(self);
            self->sock = native_traffic_open_socket(self);
            if (self->sock < 0) {
                self->error_count++;
            }
            connection_state = NATIVE_TRAFFIC_TCP_DISCONNECTED;
            next_connect_attempt_us =
                esp_timer_get_time() + NATIVE_TRAFFIC_TCP_RECONNECT_DELAY_US;
            next_send_deadline_us = 0;
            consecutive_errors = 0;
            continue;
        }

        int64_t send_started_us = esp_timer_get_time();
        ssize_t sent = native_traffic_send_packet(self, &destination);
        if (sent > 0) {
            self->packet_count++;
            consecutive_errors = 0;
        } else {
            int send_errno = errno;
            self->error_count++;
            consecutive_errors++;
            bool transient_error =
                send_errno == EAGAIN || send_errno == EWOULDBLOCK || send_errno == ENOMEM;
            if (
                (self->mode == NATIVE_TRAFFIC_MODE_DNS && !transient_error) ||
                consecutive_errors >= NATIVE_TRAFFIC_REOPEN_ERROR_COUNT
            ) {
                self->reopen_requested = true;
                consecutive_errors = 0;
            }
        }

        uint32_t rate = self->rate_pps > 0 ? self->rate_pps : 1;
        int64_t interval_us = 1000000LL / rate;
        next_send_deadline_us = native_traffic_next_send_deadline_us(
            next_send_deadline_us,
            send_started_us,
            interval_us
        );
        int64_t sleep_us = next_send_deadline_us - esp_timer_get_time();
        if (sleep_us > 0) {
            TickType_t ticks = pdMS_TO_TICKS((sleep_us + 999) / 1000);
            if (ticks > 0) {
                vTaskDelay(ticks);
            }
        }
    }

    native_traffic_close_socket(self);
    self->task = NULL;
    self->task_exited = true;
    vTaskDelete(NULL);
}

static mp_obj_t native_traffic_make_new(
    const mp_obj_type_t *type,
    size_t n_args,
    size_t n_kw,
    const mp_obj_t *args
) {
    mp_arg_check_num(n_args, n_kw, 0, 0, false);
    native_traffic_obj_t *self = mp_obj_malloc_with_finaliser(native_traffic_obj_t, type);
    self->task = NULL;
    self->running = false;
    self->paused = false;
    self->task_exited = true;
    self->reopen_requested = false;
    self->packet_count = 0;
    self->error_count = 0;
    self->gateway_addr = 0;
    self->rate_pps = 0;
    self->identifier = (uint16_t)((uintptr_t)self & 0xffff);
    self->sequence = 0;
    self->mode = NATIVE_TRAFFIC_MODE_PING;
    self->sock = -1;
    return MP_OBJ_FROM_PTR(self);
}

static mp_obj_t native_traffic_start(size_t n_args, const mp_obj_t *args) {
    native_traffic_obj_t *self = MP_OBJ_TO_PTR(args[0]);
    if (self->running || !self->task_exited) {
        mp_raise_OSError(MP_EBUSY);
    }
    const char *gateway = mp_obj_str_get_str(args[1]);
    struct in_addr address;
    if (inet_pton(AF_INET, gateway, &address) != 1) {
        mp_raise_ValueError(MP_ERROR_TEXT("invalid gateway IPv4 address"));
    }
    int rate = mp_obj_get_int(args[2]);
    if (rate <= 0 || rate > 1000) {
        mp_raise_ValueError(MP_ERROR_TEXT("rate must be 1..1000"));
    }
    native_traffic_mode_t mode = NATIVE_TRAFFIC_MODE_PING;
    if (n_args == 4) {
        const char *mode_name = mp_obj_str_get_str(args[3]);
        if (strcmp(mode_name, "dns") == 0) {
            mode = NATIVE_TRAFFIC_MODE_DNS;
        } else if (strcmp(mode_name, "ping") != 0) {
            mp_raise_ValueError(MP_ERROR_TEXT("mode must be ping or dns"));
        }
    }

    self->mode = mode;
    self->sock = native_traffic_open_socket(self);
    if (self->sock < 0) {
        mp_raise_OSError(errno > 0 ? errno : MP_EIO);
    }
    self->gateway_addr = address.s_addr;
    self->rate_pps = (uint32_t)rate;
    self->packet_count = 0;
    self->error_count = 0;
    self->sequence = 0;
    self->paused = false;
    self->reopen_requested = false;
    self->task_exited = false;
    self->running = true;
    BaseType_t result = xTaskCreate(
        native_traffic_task,
        "espectre_traffic",
        NATIVE_TRAFFIC_TASK_STACK_SIZE,
        self,
        NATIVE_TRAFFIC_TASK_PRIORITY,
        &self->task
    );
    if (result != pdPASS) {
        self->running = false;
        self->task_exited = true;
        native_traffic_close_socket(self);
        mp_raise_OSError(MP_ENOMEM);
    }
    return mp_const_true;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(
    native_traffic_start_obj,
    3,
    4,
    native_traffic_start
);

static mp_obj_t native_traffic_stop(mp_obj_t self_in) {
    native_traffic_obj_t *self = MP_OBJ_TO_PTR(self_in);
    self->running = false;
    self->paused = false;
    uint32_t waited_ms = 0;
    while (!self->task_exited && waited_ms < NATIVE_TRAFFIC_STOP_TIMEOUT_MS) {
        mp_hal_delay_ms(10);
        waited_ms += 10;
    }
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(native_traffic_stop_obj, native_traffic_stop);

static mp_obj_t native_traffic_pause(mp_obj_t self_in) {
    native_traffic_obj_t *self = MP_OBJ_TO_PTR(self_in);
    if (!self->running) {
        return mp_const_false;
    }
    self->paused = true;
    return mp_const_true;
}
static MP_DEFINE_CONST_FUN_OBJ_1(native_traffic_pause_obj, native_traffic_pause);

static mp_obj_t native_traffic_resume(mp_obj_t self_in) {
    native_traffic_obj_t *self = MP_OBJ_TO_PTR(self_in);
    if (!self->running) {
        return mp_const_false;
    }
    self->paused = false;
    return mp_const_true;
}
static MP_DEFINE_CONST_FUN_OBJ_1(native_traffic_resume_obj, native_traffic_resume);

static mp_obj_t native_traffic_reopen(mp_obj_t self_in) {
    native_traffic_obj_t *self = MP_OBJ_TO_PTR(self_in);
    if (!self->running) {
        return mp_const_false;
    }
    self->reopen_requested = true;
    return mp_const_true;
}
static MP_DEFINE_CONST_FUN_OBJ_1(native_traffic_reopen_obj, native_traffic_reopen);

static mp_obj_t native_traffic_is_running(mp_obj_t self_in) {
    native_traffic_obj_t *self = MP_OBJ_TO_PTR(self_in);
    return mp_obj_new_bool(self->running && !self->task_exited);
}
static MP_DEFINE_CONST_FUN_OBJ_1(native_traffic_is_running_obj, native_traffic_is_running);

static mp_obj_t native_traffic_packet_count(mp_obj_t self_in) {
    native_traffic_obj_t *self = MP_OBJ_TO_PTR(self_in);
    return mp_obj_new_int_from_uint(self->packet_count);
}
static MP_DEFINE_CONST_FUN_OBJ_1(native_traffic_packet_count_obj, native_traffic_packet_count);

static mp_obj_t native_traffic_error_count(mp_obj_t self_in) {
    native_traffic_obj_t *self = MP_OBJ_TO_PTR(self_in);
    return mp_obj_new_int_from_uint(self->error_count);
}
static MP_DEFINE_CONST_FUN_OBJ_1(native_traffic_error_count_obj, native_traffic_error_count);

static const mp_rom_map_elem_t native_traffic_locals_table[] = {
    { MP_ROM_QSTR(MP_QSTR_start), MP_ROM_PTR(&native_traffic_start_obj) },
    { MP_ROM_QSTR(MP_QSTR_stop), MP_ROM_PTR(&native_traffic_stop_obj) },
    { MP_ROM_QSTR(MP_QSTR_pause), MP_ROM_PTR(&native_traffic_pause_obj) },
    { MP_ROM_QSTR(MP_QSTR_resume), MP_ROM_PTR(&native_traffic_resume_obj) },
    { MP_ROM_QSTR(MP_QSTR_reopen), MP_ROM_PTR(&native_traffic_reopen_obj) },
    { MP_ROM_QSTR(MP_QSTR_is_running), MP_ROM_PTR(&native_traffic_is_running_obj) },
    { MP_ROM_QSTR(MP_QSTR_packet_count), MP_ROM_PTR(&native_traffic_packet_count_obj) },
    { MP_ROM_QSTR(MP_QSTR_error_count), MP_ROM_PTR(&native_traffic_error_count_obj) },
    { MP_ROM_QSTR(MP_QSTR___del__), MP_ROM_PTR(&native_traffic_stop_obj) },
};
static MP_DEFINE_CONST_DICT(native_traffic_locals, native_traffic_locals_table);

MP_DEFINE_CONST_OBJ_TYPE(
    native_traffic_type,
    MP_QSTR_TrafficGenerator,
    MP_TYPE_FLAG_NONE,
    make_new, native_traffic_make_new,
    locals_dict, &native_traffic_locals
);

static const mp_rom_map_elem_t native_traffic_module_globals_table[] = {
    { MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_espectre_native_traffic) },
    { MP_ROM_QSTR(MP_QSTR_TrafficGenerator), MP_ROM_PTR(&native_traffic_type) },
    // Keep the old constructor name compatible with deployed application bytecode.
    { MP_ROM_QSTR(MP_QSTR_PingGenerator), MP_ROM_PTR(&native_traffic_type) },
};
static MP_DEFINE_CONST_DICT(native_traffic_module_globals, native_traffic_module_globals_table);

const mp_obj_module_t native_traffic_module = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t *)&native_traffic_module_globals,
};

MP_REGISTER_MODULE(MP_QSTR_espectre_native_traffic, native_traffic_module);
