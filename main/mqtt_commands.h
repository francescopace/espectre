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
#include "statistics.h"
#include "calibration.h"
#include "csi_processor.h"
#include "filters.h"

// Command context - provides access to system state
typedef struct {
    runtime_config_t *config;
    float *threshold_high;
    float *threshold_low;
    stats_buffer_t *stats_buffer;
    csi_features_t *current_features;
    detection_state_t *current_state;
    butterworth_filter_t *butterworth;
    filter_buffer_t *filter_buffer;
    adaptive_normalizer_t *normalizer;
    const char *mqtt_base_topic;
    const char *mqtt_cmd_topic;
    const char *mqtt_response_topic;
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

#endif // MQTT_COMMANDS_H
