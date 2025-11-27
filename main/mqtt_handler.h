/*
 * ESPectre - MQTT Handler Module
 * 
 * Handles MQTT communication and command processing:
 * - Command dispatch and routing
 * - Response formatting and publishing
 * - Smart publishing (reduces traffic)
 * - Runtime configuration via MQTT
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#ifndef MQTT_HANDLER_H
#define MQTT_HANDLER_H

#include <stdint.h>
#include <stdbool.h>
#include "mqtt_client.h"
#include "csi_processor.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

// MQTT configuration
typedef struct {
    const char *broker_uri;
    const char *username;
    const char *password;
    const char *base_topic;
    const char *cmd_topic;
    const char *response_topic;
} mqtt_config_t;

// Smart publishing configuration
typedef struct {
    bool enabled;
    float delta_threshold;      // Minimum change to trigger publish
    float max_interval_sec;     // Maximum time between publishes (heartbeat)
} mqtt_publish_config_t;

// Motion detection result for publishing
typedef struct {
    float moving_variance;
    float threshold;
    csi_motion_state_t state;
    uint8_t segments_total;
    int64_t timestamp;
    uint32_t packets_processed;
    uint32_t packets_dropped;
    
    // Optional features (only if features_enabled && state==MOTION)
    bool has_features;
    csi_features_t features;
} segmentation_result_t;

// MQTT handler state
typedef struct {
    esp_mqtt_client_handle_t client;
    bool connected;
    
    // Smart publishing state
    float last_published_movement;
    csi_motion_state_t last_published_state;
    int64_t last_publish_time;
    bool publish_initialized;
    
    // Statistics
    uint32_t publish_count;
    uint32_t skip_count;
} mqtt_handler_state_t;

/**
 * Initialize MQTT handler
 * 
 * @param state MQTT handler state
 * @param config MQTT configuration
 * @return 0 on success, -1 on failure
 */
int mqtt_handler_init(mqtt_handler_state_t *state, const mqtt_config_t *config);

/**
 * Start MQTT client
 * 
 * @param state MQTT handler state
 * @return 0 on success, -1 on failure
 */
int mqtt_handler_start(mqtt_handler_state_t *state);

/**
 * Stop MQTT client
 * 
 * @param state MQTT handler state
 */
void mqtt_handler_stop(mqtt_handler_state_t *state);

/**
 * Check if MQTT is connected
 * 
 * @param state MQTT handler state
 * @return true if connected, false otherwise
 */
bool mqtt_handler_is_connected(const mqtt_handler_state_t *state);

/**
 * Publish segmentation result
 * 
 * @param state MQTT handler state
 * @param result Segmentation result to publish
 * @param topic Topic to publish to
 * @return 0 on success, -1 on failure
 */
int mqtt_publish_segmentation(mqtt_handler_state_t *state,
                              const segmentation_result_t *result,
                              const char *topic);


/**
 * Send response message
 * 
 * @param state MQTT handler state
 * @param message Response message
 * @param response_topic Topic to publish response to
 * @return 0 on success, -1 on failure
 */
int mqtt_send_response(mqtt_handler_state_t *state,
                      const char *message,
                      const char *response_topic);

/**
 * Check if should publish based on smart publishing rules
 * 
 * @param state MQTT handler state
 * @param current_movement Current movement score
 * @param current_state Current motion detection state
 * @param config Smart publishing configuration
 * @param current_time Current timestamp in milliseconds
 * @return true if should publish, false otherwise
 */
bool mqtt_should_publish(mqtt_handler_state_t *state,
                        float current_movement,
                        csi_motion_state_t current_state,
                        const mqtt_publish_config_t *config,
                        int64_t current_time);

/**
 * Update publish state after publishing
 * 
 * @param state MQTT handler state
 * @param movement Movement score that was published
 * @param motion_state Motion detection state that was published
 * @param current_time Current timestamp in milliseconds
 */
void mqtt_update_publish_state(mqtt_handler_state_t *state,
                              float movement,
                              csi_motion_state_t motion_state,
                              int64_t current_time);

/**
 * Set command callback for processing incoming MQTT commands
 * 
 * @param callback Function to call when command is received
 */
void mqtt_handler_set_command_callback(void (*callback)(const char *data, int data_len));

/**
 * Publish binary data to MQTT topic
 * Used for CSI raw data collection
 * 
 * @param state MQTT handler state
 * @param topic Topic to publish to
 * @param data Binary data to publish
 * @param data_len Length of binary data
 * @return 0 on success, -1 on failure
 */
int mqtt_publish_binary(mqtt_handler_state_t *state,
                       const char *topic,
                       const uint8_t *data,
                       size_t data_len);

/**
 * Context for MQTT publisher task
 */
typedef struct {
    mqtt_handler_state_t *mqtt_state;
    void *csi_processor;
    void *config;
    void *current_features;
    void *motion_state;
    void *packets_processed;
    void *packets_dropped;
    SemaphoreHandle_t state_mutex;
    const char *mqtt_topic;
} mqtt_publisher_context_t;

/**
 * Start MQTT publisher task
 * Publishes motion detection data periodically
 * 
 * @param context Publisher context with all required state
 */
void mqtt_start_publisher(mqtt_publisher_context_t *context);

#endif // MQTT_HANDLER_H
