/*
 * ESPectre - Test Harness
 *
 * Minimal host-side test harness helpers for native C++ suites.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace espectre::test {

struct AssertionFailure : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct TestSkipped : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

int begin_suite();
void run_test(const char *name, void (*fn)());
int end_suite();

[[noreturn]] void fail(const char *file, int line, const std::string &message);
[[noreturn]] void skip(const char *file, int line, const std::string &message);

template <typename T>
std::string render_value(const T &value) {
  std::ostringstream oss;
  if constexpr (std::is_same_v<std::decay_t<T>, bool>) {
    oss << (value ? "true" : "false");
  } else if constexpr (std::is_enum_v<std::decay_t<T>>) {
    using Underlying = std::underlying_type_t<std::decay_t<T>>;
    oss << static_cast<Underlying>(value);
  } else if constexpr (std::is_pointer_v<std::decay_t<T>>) {
    oss << static_cast<const void *>(value);
  } else {
    oss << value;
  }
  return oss.str();
}

inline void assert_true(bool condition, const char *expression, const char *file, int line,
                        const char *message = nullptr) {
  if (condition) {
    return;
  }

  std::ostringstream oss;
  oss << "Expected true: " << expression;
  if (message != nullptr) {
    oss << " (" << message << ")";
  }
  fail(file, line, oss.str());
}

inline void assert_false(bool condition, const char *expression, const char *file, int line,
                         const char *message = nullptr) {
  if (!condition) {
    return;
  }

  std::ostringstream oss;
  oss << "Expected false: " << expression;
  if (message != nullptr) {
    oss << " (" << message << ")";
  }
  fail(file, line, oss.str());
}

namespace detail {

// C++17 equivalent of std::cmp_equal: mixed-signed integer asserts must not
// emit -Wsign-compare, and a negative value must not compare equal to a large
// unsigned wraparound.
template <typename T, typename U>
constexpr bool values_equal(const T &lhs, const U &rhs) {
  using Left = std::decay_t<T>;
  using Right = std::decay_t<U>;
  if constexpr (std::is_integral_v<Left> && std::is_integral_v<Right>) {
    if constexpr (std::is_signed_v<Left> == std::is_signed_v<Right>) {
      return lhs == rhs;
    } else if constexpr (std::is_signed_v<Left>) {
      return lhs < 0 ? false : static_cast<std::make_unsigned_t<Left>>(lhs) == rhs;
    } else {
      return rhs < 0 ? false : lhs == static_cast<std::make_unsigned_t<Right>>(rhs);
    }
  } else {
    return lhs == rhs;
  }
}

}  // namespace detail

template <typename Expected, typename Actual>
inline void assert_equal(const Expected &expected, const Actual &actual, const char *file, int line,
                         const char *message = nullptr) {
  if (detail::values_equal(expected, actual)) {
    return;
  }

  std::ostringstream oss;
  oss << "Expected " << render_value(expected) << " but got " << render_value(actual);
  if (message != nullptr) {
    oss << " (" << message << ")";
  }
  fail(file, line, oss.str());
}

inline void assert_equal_string(const char *expected, const char *actual, const char *file, int line,
                                const char *message = nullptr) {
  if (expected != nullptr && actual != nullptr && std::string(expected) == actual) {
    return;
  }
  if (expected == nullptr && actual == nullptr) {
    return;
  }

  std::ostringstream oss;
  oss << "Expected string " << render_value(expected) << " but got " << render_value(actual);
  if (message != nullptr) {
    oss << " (" << message << ")";
  }
  fail(file, line, oss.str());
}

inline void assert_equal_float(float expected, float actual, const char *file, int line,
                               const char *message = nullptr) {
  if (std::fabs(expected - actual) <= 1e-6f) {
    return;
  }

  std::ostringstream oss;
  oss << "Expected float " << expected << " but got " << actual;
  if (message != nullptr) {
    oss << " (" << message << ")";
  }
  fail(file, line, oss.str());
}

inline void assert_float_within(float delta, float expected, float actual, const char *file, int line,
                                const char *message = nullptr) {
  if (std::fabs(expected - actual) <= delta) {
    return;
  }

  std::ostringstream oss;
  oss << "Expected float within " << delta << " of " << expected << " but got " << actual;
  if (message != nullptr) {
    oss << " (" << message << ")";
  }
  fail(file, line, oss.str());
}

// Comparing a single-precision runtime value against a double-precision
// reference needs the reference to stay in double, so the tolerance cannot be
// narrowed to float on the way in.
inline void assert_double_within(double delta, double expected, double actual, const char *file,
                                 int line, const char *message = nullptr) {
  if (std::fabs(expected - actual) <= delta) {
    return;
  }

  std::ostringstream oss;
  oss.precision(12);
  oss << "Expected double within " << delta << " of " << expected << " but got " << actual;
  if (message != nullptr) {
    oss << " (" << message << ")";
  }
  fail(file, line, oss.str());
}

inline void assert_null(const void *value, const char *file, int line, const char *message = nullptr) {
  if (value == nullptr) {
    return;
  }

  std::ostringstream oss;
  oss << "Expected null but got " << render_value(value);
  if (message != nullptr) {
    oss << " (" << message << ")";
  }
  fail(file, line, oss.str());
}

inline void assert_not_null(const void *value, const char *file, int line,
                            const char *message = nullptr) {
  if (value != nullptr) {
    return;
  }

  std::ostringstream oss;
  oss << "Expected non-null value";
  if (message != nullptr) {
    oss << " (" << message << ")";
  }
  fail(file, line, oss.str());
}

template <typename T>
inline void assert_equal_array(const T *expected, const T *actual, std::size_t size,
                               const char *file, int line, const char *message = nullptr) {
  for (std::size_t i = 0; i < size; ++i) {
    if (expected[i] == actual[i]) {
      continue;
    }

    std::ostringstream oss;
    oss << "Array mismatch at index " << i << ": expected " << render_value(expected[i])
        << " but got " << render_value(actual[i]);
    if (message != nullptr) {
      oss << " (" << message << ")";
    }
    fail(file, line, oss.str());
  }
}

}  // namespace espectre::test

#define UNITY_BEGIN() ::espectre::test::begin_suite()
#define UNITY_END() ::espectre::test::end_suite()
#define RUN_TEST(fn) ::espectre::test::run_test(#fn, fn)

#define TEST_ASSERT_TRUE(condition) \
  ::espectre::test::assert_true((condition), #condition, __FILE__, __LINE__)

#define TEST_ASSERT_TRUE_MESSAGE(condition, message) \
  ::espectre::test::assert_true((condition), #condition, __FILE__, __LINE__, (message))

#define TEST_ASSERT_FALSE(condition) \
  ::espectre::test::assert_false((condition), #condition, __FILE__, __LINE__)

#define TEST_ASSERT_FALSE_MESSAGE(condition, message) \
  ::espectre::test::assert_false((condition), #condition, __FILE__, __LINE__, (message))

#define TEST_ASSERT_EQUAL(expected, actual) \
  ::espectre::test::assert_equal((expected), (actual), __FILE__, __LINE__)

#define TEST_ASSERT_EQUAL_MESSAGE(expected, actual, message) \
  ::espectre::test::assert_equal((expected), (actual), __FILE__, __LINE__, (message))

#define TEST_ASSERT_EQUAL_INT_MESSAGE(expected, actual, message) \
  ::espectre::test::assert_equal((expected), (actual), __FILE__, __LINE__, (message))

#define TEST_ASSERT_EQUAL_INT(expected, actual) \
  ::espectre::test::assert_equal((expected), (actual), __FILE__, __LINE__)

#define TEST_ASSERT_EQUAL_UINT8(expected, actual) \
  ::espectre::test::assert_equal(static_cast<std::uint8_t>(expected), static_cast<std::uint8_t>(actual), __FILE__, __LINE__)

#define TEST_ASSERT_EQUAL_UINT8_MESSAGE(expected, actual, message) \
  ::espectre::test::assert_equal(static_cast<std::uint8_t>(expected), static_cast<std::uint8_t>(actual), __FILE__, __LINE__, (message))

#define TEST_ASSERT_EQUAL_INT8(expected, actual) \
  ::espectre::test::assert_equal(static_cast<std::int8_t>(expected), static_cast<std::int8_t>(actual), __FILE__, __LINE__)

#define TEST_ASSERT_EQUAL_UINT8_ARRAY(expected, actual, size) \
  ::espectre::test::assert_equal_array((expected), (actual), static_cast<std::size_t>(size), __FILE__, __LINE__)

#define TEST_ASSERT_EQUAL_FLOAT(expected, actual) \
  ::espectre::test::assert_equal_float((expected), (actual), __FILE__, __LINE__)

#define TEST_ASSERT_FLOAT_WITHIN(delta, expected, actual) \
  ::espectre::test::assert_float_within((delta), (expected), (actual), __FILE__, __LINE__)

#define TEST_ASSERT_DOUBLE_WITHIN(delta, expected, actual) \
  ::espectre::test::assert_double_within((delta), (expected), (actual), __FILE__, __LINE__)

#define TEST_ASSERT_EQUAL_STRING(expected, actual) \
  ::espectre::test::assert_equal_string((expected), (actual), __FILE__, __LINE__)

#define TEST_ASSERT_NULL(value) \
  ::espectre::test::assert_null((value), __FILE__, __LINE__)

#define TEST_ASSERT_NOT_NULL(value) \
  ::espectre::test::assert_not_null((value), __FILE__, __LINE__)

#define TEST_ASSERT_NOT_NULL_MESSAGE(value, message) \
  ::espectre::test::assert_not_null((value), __FILE__, __LINE__, (message))

#define TEST_IGNORE_MESSAGE(message) \
  ::espectre::test::skip(__FILE__, __LINE__, (message))

#define TEST_PASS() \
  do {             \
    return;        \
  } while (false)
