/*
 * ESPectre - Utility Functions
 * 
 * Shared utility functions used across multiple modules.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include <cstdint>

namespace esphome {
namespace espectre {

/**
 * Compare two float values for qsort
 * 
 * @param a Pointer to first float
 * @param b Pointer to second float
 * @return -1 if a < b, 0 if a == b, 1 if a > b
 */
inline int compare_float(const void *a, const void *b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

/**
 * Compare two int8_t values for qsort
 * 
 * @param a Pointer to first int8_t
 * @param b Pointer to second int8_t
 * @return Difference between values
 */
inline int compare_int8(const void *a, const void *b) {
    return (*(const int8_t*)a - *(const int8_t*)b);
}

/**
 * Compare absolute values of two floats for qsort
 * 
 * @param a Pointer to first float
 * @param b Pointer to second float
 * @return -1 if |a| < |b|, 0 if |a| == |b|, 1 if |a| > |b|
 */
inline int compare_float_abs(const void *a, const void *b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    if (fa < 0) fa = -fa;
    if (fb < 0) fb = -fb;
    return (fa > fb) - (fa < fb);
}

}  // namespace espectre
}  // namespace esphome
