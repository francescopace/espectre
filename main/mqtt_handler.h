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
#include "detection_engine.h"

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

// MQTT handler state
typedef struct {
    esp_mqtt_client_handle_t client;
    bool connected;
    
    // Smart publishing state
    float last_published_movement;
    detection_state_t last_published_state;
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
 * Publish detection result
 * 
 * @param state MQTT handler state
 * @param result Detection result to publish
 * @param topic Topic to publish to
 * @return 0 on success, -1 on failure
 */
int mqtt_publish_detection(mqtt_handler_state_t *state,
                           const detection_result_t *result,
                           const char *topic);

/**
 * Publish calibration status update
 * 
 * @param state MQTT handler state
 * @param phase Current calibration phase
 * @param phase_duration Duration of the phase in seconds
 * @param samples_collected Number of samples collected
 * @param topic Topic to publish to
 * @return 0 on success, -1 on failure
 */
int mqtt_publish_calibration_status(mqtt_handler_state_t *state,
                                    uint8_t phase,
                                    uint32_t phase_target_samples,
                                    uint32_t samples_collected,
                                    uint32_t traffic_rate,
                                    const char *topic);

/**
 * Publish calibration complete recap with detailed summary
 * 
 * @param state MQTT handler state
 * @param calib_results Calibration results (void* to calibration_state_t)
 * @param topic Topic to publish to
 * @return 0 on success, -1 on failure
 */
int mqtt_publish_calibration_complete(mqtt_handler_state_t *state,
                                      const void *calib_results,
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
 * @param current_state Current detection state
 * @param config Smart publishing configuration
 * @param current_time Current timestamp in milliseconds
 * @return true if should publish, false otherwise
 */
bool mqtt_should_publish(mqtt_handler_state_t *state,
                        float current_movement,
                        detection_state_t current_state,
                        const mqtt_publish_config_t *config,
                        int64_t current_time);

/**
 * Update publish state after publishing
 * 
 * @param state MQTT handler state
 * @param movement Movement score that was published
 * @param det_state Detection state that was published
 * @param current_time Current timestamp in milliseconds
 */
void mqtt_update_publish_state(mqtt_handler_state_t *state,
                              float movement,
                              detection_state_t det_state,
                              int64_t current_time);

/**
 * Get publish statistics
 * 
 * @param state MQTT handler state
 * @param published Output: number of messages published
 * @param skipped Output: number of messages skipped
 */
void mqtt_get_publish_stats(const mqtt_handler_state_t *state,
                           uint32_t *published,
                           uint32_t *skipped);

/**
 * Set command callback for processing incoming MQTT commands
 * 
 * @param callback Function to call when command is received
 */
void mqtt_handler_set_command_callback(void (*callback)(const char *data, int data_len));

/**
 * Publish CSI raw data batch
 * 
 * @param batch Array of CSI packets (100 packets x 128 bytes)
 * @param count Number of packets in batch
 * @param phase Current calibration phase
 */
void mqtt_publish_csi_batch(const int8_t batch[][128], uint32_t count, uint8_t phase);

#endif // MQTT_HANDLER_H
