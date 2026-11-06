/*
 * ESPectre - Mock driver/uart.h
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>

#include "esp_err.h"
#include "freertos/FreeRTOS.h"

typedef int uart_port_t;
constexpr uart_port_t UART_NUM_1 = 1;

enum uart_word_length_t { UART_DATA_8_BITS = 3 };
enum uart_parity_t { UART_PARITY_DISABLE = 0 };
enum uart_stop_bits_t { UART_STOP_BITS_1 = 1 };
enum uart_hw_flowcontrol_t { UART_HW_FLOWCTRL_DISABLE = 0 };
enum uart_sclk_t { UART_SCLK_DEFAULT = 0 };

struct uart_config_t {
  int baud_rate;
  uart_word_length_t data_bits;
  uart_parity_t parity;
  uart_stop_bits_t stop_bits;
  uart_hw_flowcontrol_t flow_ctrl;
  uint8_t rx_flow_ctrl_thresh;
  uart_sclk_t source_clk;
  struct {
    uint32_t allow_pd : 1;
    uint32_t backup_before_sleep : 1;
  } flags;
};

struct uart_mock_state_t {
  int driver_install_count{0};
};

inline uart_mock_state_t g_uart_mock{};

inline esp_err_t uart_driver_install(uart_port_t, int, int, int, void *, int) {
  g_uart_mock.driver_install_count++;
  return ESP_OK;
}

inline esp_err_t uart_param_config(uart_port_t, const uart_config_t *) { return ESP_OK; }

inline int uart_read_bytes(uart_port_t, void *, uint32_t, TickType_t) { return 0; }
