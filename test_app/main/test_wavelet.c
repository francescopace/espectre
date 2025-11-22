/*
 * ESPectre - Wavelet Filter Unit Tests
 *
 * Tests for Daubechies db4 wavelet transform
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "test_case_esp.h"
#include "wavelet.h"
#include <math.h>
#include <string.h>

// Test: Wavelet initialization
TEST_CASE_ESP(wavelet_filter_initialization, "[wavelet]") {
  wavelet_state_t state;
  wavelet_init(&state, 3, 1.0f, WAVELET_THRESH_SOFT);

  TEST_ASSERT_TRUE(state.initialized);
  TEST_ASSERT_EQUAL(3, state.decomp_level);
  TEST_ASSERT_FLOAT_WITHIN(0.01f, 1.0f, state.threshold);
  TEST_ASSERT_EQUAL(WAVELET_THRESH_SOFT, state.method);
  TEST_ASSERT_EQUAL(0, state.buffer_count);
}

// Test: Soft thresholding
TEST_CASE_ESP(wavelet_soft_thresholding, "[wavelet]") {
  float threshold = 1.0f;

  // Value above threshold
  float result1 = wavelet_soft_threshold(2.5f, threshold);
  TEST_ASSERT_FLOAT_WITHIN(0.01f, 1.5f, result1); // 2.5 - 1.0 = 1.5

  // Value below threshold
  float result2 = wavelet_soft_threshold(0.5f, threshold);
  TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.0f, result2); // Should be 0

  // Negative value above threshold
  float result3 = wavelet_soft_threshold(-2.5f, threshold);
  TEST_ASSERT_FLOAT_WITHIN(0.01f, -1.5f, result3); // -(2.5 - 1.0) = -1.5
}

// Test: Hard thresholding
TEST_CASE_ESP(wavelet_hard_thresholding, "[wavelet]") {
  float threshold = 1.0f;

  // Value above threshold
  float result1 = wavelet_hard_threshold(2.5f, threshold);
  TEST_ASSERT_FLOAT_WITHIN(0.01f, 2.5f, result1); // Keep original

  // Value below threshold
  float result2 = wavelet_hard_threshold(0.5f, threshold);
  TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.0f, result2); // Should be 0

  // Negative value above threshold
  float result3 = wavelet_hard_threshold(-2.5f, threshold);
  TEST_ASSERT_FLOAT_WITHIN(0.01f, -2.5f, result3); // Keep original
}

// Test: Wavelet denoising on synthetic signal
TEST_CASE_ESP(wavelet_denoising_reduces_noise, "[wavelet]") {
  // Create a clean signal (sine wave) + significant noise
  const size_t length = 32; // Must be power of 2
  float input[length];
  float output[length];

  // Generate sine wave with more significant noise
  for (size_t i = 0; i < length; i++) {
    float clean = sinf(2.0f * M_PI * i / length);
    float noise = (i % 2 == 0) ? 2.0f : -2.0f; // Alternating noise
    input[i] = clean + noise;
  }

  // Apply wavelet denoising with higher threshold
  int result =
      wavelet_denoise(input, output, length, 2, 1.5f, WAVELET_THRESH_SOFT);

  TEST_ASSERT_EQUAL(0, result);

  // Output should be smoother (less variance)
  float input_var = 0.0f, output_var = 0.0f;
  float input_mean = 0.0f, output_mean = 0.0f;

  for (size_t i = 0; i < length; i++) {
    input_mean += input[i];
    output_mean += output[i];
  }
  input_mean /= length;
  output_mean /= length;

  for (size_t i = 0; i < length; i++) {
    input_var += (input[i] - input_mean) * (input[i] - input_mean);
    output_var += (output[i] - output_mean) * (output[i] - output_mean);
  }
  input_var /= length;
  output_var /= length;

  // Denoised signal should have lower or similar variance (wavelet removes
  // high-frequency noise)
  TEST_ASSERT_LESS_OR_EQUAL(input_var, output_var);
}

// Test: Streaming mode with circular buffer
TEST_CASE_ESP(wavelet_streaming_mode, "[wavelet]") {
  wavelet_state_t state;
  wavelet_init(&state, 2, 1.0f, WAVELET_THRESH_SOFT);

  // Feed samples until buffer is full
  for (int i = 0; i < WAVELET_BUFFER_SIZE + 5; i++) {
    float input = sinf(2.0f * M_PI * i / 16.0f) + ((i % 5 == 0) ? 0.3f : 0.0f);
    float output = wavelet_denoise_sample(&state, input);

    // After buffer fills, output should be valid
    if (i >= WAVELET_BUFFER_SIZE) {
      TEST_ASSERT_FALSE(isnan(output));
      TEST_ASSERT_FALSE(isinf(output));
    }
  }

  TEST_ASSERT_EQUAL(WAVELET_BUFFER_SIZE, state.buffer_count);
}

// Test: Wavelet with invalid parameters
TEST_CASE_ESP(wavelet_handles_invalid_parameters, "[wavelet]") {
  float input[32];
  float output[32];

  // NULL pointers
  int result1 = wavelet_denoise(NULL, output, 32, 2, 1.0f, WAVELET_THRESH_SOFT);
  TEST_ASSERT_EQUAL(-1, result1);

  int result2 = wavelet_denoise(input, NULL, 32, 2, 1.0f, WAVELET_THRESH_SOFT);
  TEST_ASSERT_EQUAL(-1, result2);

  // Signal too short
  int result3 = wavelet_denoise(input, output, 4, 3, 1.0f, WAVELET_THRESH_SOFT);
  TEST_ASSERT_EQUAL(-1, result3);
}

// Test: Wavelet level clamping
TEST_CASE_ESP(wavelet_level_clamping, "[wavelet]") {
  wavelet_state_t state;

  // Level too high (should clamp to 3)
  wavelet_init(&state, 5, 1.0f, WAVELET_THRESH_SOFT);
  TEST_ASSERT_EQUAL(3, state.decomp_level);

  // Level too low (should clamp to 1)
  wavelet_init(&state, 0, 1.0f, WAVELET_THRESH_SOFT);
  TEST_ASSERT_EQUAL(1, state.decomp_level);

  // Valid level
  wavelet_init(&state, 2, 1.0f, WAVELET_THRESH_SOFT);
  TEST_ASSERT_EQUAL(2, state.decomp_level);
}

// Test: Wavelet coefficients are normalized
TEST_CASE_ESP(wavelet_db4_coefficients_sum_correctly, "[wavelet]") {
  // Low-pass filter should sum to sqrt(2) for normalization
  float lp_sum = 0.0f;
  for (int i = 0; i < WAVELET_DB4_LENGTH; i++) {
    lp_sum += WAVELET_DB4_LP[i];
  }
  TEST_ASSERT_FLOAT_WITHIN(0.01f, sqrtf(2.0f), lp_sum);

  // High-pass filter should sum to 0 (alternating signs)
  float hp_sum = 0.0f;
  for (int i = 0; i < WAVELET_DB4_LENGTH; i++) {
    hp_sum += WAVELET_DB4_HP[i];
  }
  TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.0f, hp_sum);
}

// Test: Wavelet preserves DC component
TEST_CASE_ESP(wavelet_preserves_dc_component, "[wavelet]") {
  const size_t length = 32;
  float input[length];
  float output[length];

  // Constant signal (DC component)
  for (size_t i = 0; i < length; i++) {
    input[i] = 5.0f;
  }

  wavelet_denoise(input, output, length, 2, 0.5f, WAVELET_THRESH_SOFT);

  // Mean should be preserved
  float output_mean = 0.0f;
  for (size_t i = 0; i < length; i++) {
    output_mean += output[i];
  }
  output_mean /= length;

  TEST_ASSERT_FLOAT_WITHIN(0.5f, 5.0f, output_mean);
}

// Test: Wavelet allocation failure protection
TEST_CASE_ESP(wavelet_allocation_failure_protection, "[wavelet]") {
  float *input = NULL;
  float *output = NULL;

  // Test 1: Overflow protection
  // We can't easily allocate SIZE_MAX bytes to test the actual overflow,
  // but we can test that the function handles the check safely.
  // However, passing a huge size might cause other issues if the check fails.
  // The check is: if (max_size == 0 || max_size > SIZE_MAX / sizeof(float))

  // Case A: Zero size
  int result_zero =
      wavelet_denoise(input, output, 0, 2, 0.5f, WAVELET_THRESH_SOFT);
  TEST_ASSERT_EQUAL(-1, result_zero);

  // Case B: Potential overflow size (simulated by passing a very large number
  // that would fail the check) Note: We can't pass a real buffer of this size,
  // but the function checks size before accessing input/output for the
  // allocation. However, it DOES check input/output for NULL first. So we need
  // valid pointers but invalid size.
  float dummy_in[1];
  float dummy_out[1];

  // We use a size that will trigger the overflow check.
  // SIZE_MAX / sizeof(float) + 1 would wrap around if not checked, or be just
  // too large. Since we can't easily construct a size that definitely overflows
  // size_t (since it IS size_t), we rely on the fact that any size larger than
  // available memory would fail malloc/calloc. But we specifically added a
  // check for > SIZE_MAX / sizeof(float). Let's try to pass a size that is
  // technically valid as size_t but would cause overflow in the multiplication.
  // max_size * sizeof(float).
  // If size_t is 32-bit, max is 4GB. sizeof(float) is 4. So max_size > 1GB
  // triggers it.

  size_t huge_size = (SIZE_MAX / sizeof(float)) + 1;
  // Note: This calculation itself might wrap if not careful, but here we are
  // just producing a value. Actually, (SIZE_MAX / 4) + 1 is a valid size_t
  // value.

  int result_overflow = wavelet_denoise(dummy_in, dummy_out, huge_size, 2, 0.5f,
                                        WAVELET_THRESH_SOFT);
  TEST_ASSERT_EQUAL(-1, result_overflow);
}
