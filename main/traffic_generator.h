/*
 * ESPectre - WiFi Traffic Generator
 * 
 * Generates continuous UDP broadcast packets to ensure CSI data availability.
 * Essential for CSI-based detection as ESP32 only receives CSI when there's WiFi traffic.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#ifndef TRAFFIC_GENERATOR_H
#define TRAFFIC_GENERATOR_H

#include <stdint.h>
#include <stdbool.h>
#include "espectre.h"

// Initialize traffic generator
void traffic_generator_init(void);

// Start traffic generator with specified rate (packets per second)
bool traffic_generator_start(uint32_t rate_pps);

// Stop traffic generator
void traffic_generator_stop(void);

// Check if traffic generator is running
bool traffic_generator_is_running(void);

// Get statistics
uint32_t traffic_generator_get_packet_count(void);

// Set rate (packets per second) while running
void traffic_generator_set_rate(uint32_t rate_pps);

#endif // TRAFFIC_GENERATOR_H
