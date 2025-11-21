/*
 * ESPectre - MQTT Commands Module
 * 
 * Handles MQTT command processing and execution:
 * - Command parsing and validation
 * - Runtime parameter updates
 * - System control commands
 * - Response generation
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#ifndef MQTT_COMMANDS_H
#define MQTT_COMMANDS_H

#include "mqtt_handler.h"
#include "config_manager.h"
#include "csi_processor.h"
#include "filters.h"
#include "segmentation.h"

// Command context - provides access to system state
typedef struct {
    runtime_config_t *config;
    csi_features_t *current_features;
    segmentation_state_t *current_state;
    butterworth_filter_t *butterworth;
    filter_buffer_t *filter_buffer;
    segmentation_context_t *segmentation;
    const char *mqtt_base_topic;
    const char *mqtt_cmd_topic;
    const char *mqtt_response_topic;
    int64_t *system_start_time;  // Pointer to system start timestamp (for uptime calculation)
} mqtt_cmd_context_t;

/**
 * Initialize MQTT command handler
 * 
 * @param mqtt_state MQTT handler state
 * @param context Command context with system state
 * @return 0 on success, -1 on failure
 */
int mqtt_commands_init(mqtt_handler_state_t *mqtt_state, mqtt_cmd_context_t *context);

/**
 * Process incoming MQTT command
 * 
 * @param data Command data (JSON string)
 * @param data_len Length of command data
 * @param context Command context
 * @param mqtt_state MQTT handler state for responses
 * @param response_topic Topic to send responses to
 * @return 0 on success, -1 on failure
 */
int mqtt_commands_process(const char *data, int data_len, 
                         mqtt_cmd_context_t *context,
                         mqtt_handler_state_t *mqtt_state,
                         const char *response_topic);

/**
 * Capture CSI raw packet for data collection
 * Called from csi_callback when raw capture is active
 * 
 * @param csi_data Raw CSI data (128 bytes)
 * @param csi_len Length of CSI data
 */
void mqtt_commands_capture_csi_packet(const int8_t *csi_data, size_t csi_len);

#endif // MQTT_COMMANDS_H
