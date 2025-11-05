/*
 * Custom TEST_CASE macro for ESP-IDF that exports test descriptors
 * This is needed because the standard TEST_CASE uses static storage
 * and constructor attributes which don't work properly in ESP-IDF
 */

#ifndef TEST_CASE_ESP_H
#define TEST_CASE_ESP_H

#include "unity.h"
#include "unity_test_runner.h"

// Custom TEST_CASE that makes descriptors non-static and globally visible
#define TEST_CASE_ESP(name_, desc_) \
    static void UNITY_TEST_UID(test_func_) (void); \
    test_desc_t UNITY_TEST_UID(test_desc_) = { \
        .name = name_, \
        .desc = desc_, \
        .fn = (test_func[]){&UNITY_TEST_UID(test_func_)}, \
        .file = __FILE__, \
        .line = __LINE__, \
        .test_fn_count = 1, \
        .test_fn_name = NULL, \
        .next = NULL \
    }; \
    static void UNITY_TEST_UID(test_func_) (void)

#endif // TEST_CASE_ESP_H
