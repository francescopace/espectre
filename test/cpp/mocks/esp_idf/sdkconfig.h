/*
 * ESPectre - Mock sdkconfig.h
 *
 * Host-side mock of sdkconfig.h for native C++ tests.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#ifndef SDKCONFIG_H
#define SDKCONFIG_H

// Mock ESP32 SDK configuration

// FreeRTOS configuration
#define CONFIG_FREERTOS_HZ 100
#define CONFIG_FREERTOS_UNICORE 0

// ESP32 chip configuration
#define CONFIG_IDF_TARGET_ESP32 1
#define CONFIG_IDF_TARGET "esp32"

// WiFi configuration
#define CONFIG_ESP32_WIFI_ENABLED 1
#define CONFIG_ESP32_WIFI_STATIC_RX_BUFFER_NUM 10
#define CONFIG_ESP32_WIFI_DYNAMIC_RX_BUFFER_NUM 32
#define CONFIG_ESP32_WIFI_TX_BUFFER_TYPE 1
#define CONFIG_ESP32_WIFI_DYNAMIC_TX_BUFFER_NUM 32
#define CONFIG_ESP32_WIFI_CSI_ENABLED 1
#define CONFIG_ESPECTRE_WIFI_SSID ""
#define CONFIG_ESPECTRE_WIFI_PASSWORD ""
#define CONFIG_ESPECTRE_WIFI_BSSID ""
#define CONFIG_ESPECTRE_WIFI_CHANNEL 0

// Memory configuration
#define CONFIG_ESP32_DEFAULT_CPU_FREQ_MHZ 240
#define CONFIG_SPIRAM_SUPPORT 0

// Logging
#define CONFIG_LOG_DEFAULT_LEVEL 3
#define CONFIG_LOG_MAXIMUM_LEVEL 5

#endif // SDKCONFIG_H
