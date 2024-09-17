#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cctype>
#include <chrono>
#include <cinttypes>
#include <codecvt>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <list>
#include <locale>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <queue>
#include <random>
#include <regex>
#include <set>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "lab/common/forwards.h"
#include "lab/common/platformdetect.h"
#include "lab/common/version.h"

#ifdef LAB_DEBUG
#ifdef LAB_PLATFORM_WINDOWS
#define LAB_BREAK() __debugbreak()
#else
#include <csignal>
#define LAB_BREAK() raise(SIGTRAP)
#endif
#else
#define LAB_BREAK() ((void)0)
#endif

// enable log
// Core log macros
#ifdef LAB_DEBUG
#define LAB_ENABLE_LOG
#define LAB_ENABLE_CHECK
#endif // LAB_DEBUG

#define LAB_NODISCARD [[nodiscard]]

#ifdef LAB_ENABLE_LOG
#include "lab/common/logger.h"

#define LAB_LOG(logger, severity, msg)   \
  {                                      \
    std::stringstream sstream{};         \
    sstream << msg;                      \
    logger.log(severity, sstream.str()); \
  }

#define LAB_LOG_VERBOSE(msg) LAB_LOG(::lab::get_logger(), ::lab::Logger::Severity::VERBOSE, msg)
#define LAB_LOG_DEBUG(msg) LAB_LOG(::lab::get_logger(), ::lab::Logger::Severity::DEBUG, msg)
#define LAB_LOG_INFO(msg) LAB_LOG(::lab::get_logger(), ::lab::Logger::Severity::INFO, msg)
#define LAB_LOG_WARN(msg) LAB_LOG(::lab::get_logger(), ::lab::Logger::Severity::WARNING, msg)
#define LAB_LOG_ERROR(msg) LAB_LOG(::lab::get_logger(), ::lab::Logger::Severity::ERROR, msg)
#define LAB_LOG_FATAL(msg) LAB_LOG(::lab::get_logger(), ::lab::Logger::Severity::FATAL, msg)

#else
#define LAB_LOG_VERBOSE(...) ((void)0)
#define LAB_LOG_DEBUG(...) ((void)0)
#define LAB_LOG_INFO(...) ((void)0)
#define LAB_LOG_WARN(...) ((void)0)
#define LAB_LOG_ERROR(...) ((void)0)
#define LAB_LOG_FATAL(...) ((void)0)
#endif

#define LAB_UNIMPLEMENTED LAB_LOG_FATAL("Unimplemented in " << __FILE__ << "(" << __LINE__ << ")");
#define LAB_UNREACHABLE LAB_LOG_FATAL("Unexpected in " << __FILE__ << "(" << __LINE__ << ")")

#ifdef LAB_PLATFORM_WINDOWS
#pragma warning(disable : 4251)
#ifdef LAB_DYNAMIC
#ifdef LAB_ENGINE
#define LAB_EXPORT __declspec(dllexport)
#else
#define LAB_EXPORT __declspec(dllimport)
#endif
#else
#define LAB_EXPORT
#endif
#define LAB_HIDDEN
#else
#define LAB_EXPORT __attribute__((visibility("default")))
#define LAB_HIDDEN __attribute__((visibility("hidden")))
#endif

#if (defined(_CPPUNWIND) || defined(__EXCEPTIONS))
#define LAB_EXCEPTIONS
#endif

#if defined(_MSC_VER)
#define LAB_IS_MSCV
#define DISABLE_WARNING_PUSH __pragma(warning(push))
#define DISABLE_WARNING_POP __pragma(warning(pop))
#define DISABLE_WARNING(warningNumber) __pragma(warning(disable : warningNumber))
#define DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER DISABLE_WARNING(4100)
#define DISABLE_WARNING_UNREFERENCED_FUNCTION DISABLE_WARNING(4505)
#define DISABLE_WARNING_CONVERSION_TO_SMALLER_TYPE DISABLE_WARNING(4267)
#else
#define DISABLE_WARNING_PUSH
#define DISABLE_WARNING_POP
#define DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER
#define DISABLE_WARNING_UNREFERENCED_FUNCTION
#define DISABLE_WARNING_CONVERSION_TO_SMALLER_TYPE
#endif

#ifndef LAB_FORCE_INLINE
#if defined(_MSC_VER)
#define LAB_FORCE_INLINE __forceinline
#else
#define LAB_FORCE_INLINE inline
#endif
#endif // !LAB_FORCE_INLINE

#ifndef LAB_ALWAYS_INLINE
#if defined(_MSC_VER)
#define LAB_ALWAYS_INLINE LAB_FORCE_INLINE
#else
#define LAB_ALWAYS_INLINE inline
#endif
#endif // !LAB_ALWAYS_INLINE

#ifndef LAB_NODISCARD
#define LAB_NODISCARD [[nodiscard]]
#endif // !LAB_NODISCARD

#ifndef LAB_ALLOW_DISCARD
#define LAB_ALLOW_DISCARD (void)
#endif // !LAB_ALLOW_DISCARD

#if defined(_MSC_VER)
// NOTE MSVC often gives C4127 warnings with compiletime if statements. See bug
// 1362. This workaround is ugly, but it does the job.
#define LAB_CONST_CONDITIONAL(cond) (void)0, cond
#else
#define LAB_CONST_CONDITIONAL(cond) cond
#endif

#define LAB_THROW(msg)         \
  std::stringstream sstream{}; \
  sstream << msg;              \
  throw std::runtime_error {   \
    sstream.str()              \
  }

#define LAB_ASSERT(cond)                                                            \
  if (!(!(!(cond)))) {                                                              \
    LAB_THROW(#cond << " Assert failed at " << __FILE__ << "(" << __LINE__ << ")"); \
  }

#ifdef LAB_ENABLE_CHECK
#define LAB_CHECK(cond, msg)                                                                            \
  if (!(!(!(cond)))) {                                                                                  \
    LAB_THROW("Check " << #cond << " failed: " << msg << " at " << __FILE__ << "(" << __LINE__ << ")"); \
  }
#else
#define EMPTY_CHECK \
  do {              \
  } while (false);

#define LAB_CHECK(cond, msg) EMPTY_CHECK
#endif // LAB_ENABLE_CHECK

#ifndef LAB_MAX_RECURSION
#define LAB_MAX_RECURSION 100
#endif // LAB_MAX_RECURSION

#ifndef LAB_STR
#define LAB_STR(x) #x
#define LAB_MAKE_STR(x) STR(x)
#endif // !LAB_STR

#ifndef BIT
#define BIT(x) (1 << x)
#endif // !BIT

#ifndef LAB_SHIFT_LEFT
#define LAB_SHIFT_LEFT(x) (std::size(1) << x)
#endif //

#ifndef LAB_SHIFT_RIGHT
#define LAB_SHIFT_RIGHT(x) (std::size(1) >> x)
#endif // !LAB_SHIFT_RIGHT

#ifndef LAB_CONCAT
#define LAB_CONCAT_HELPER(x, y) x##y
#define LAB_CONCAT(x, y) LAB_CONCAT_HELPER(x, y)
#endif // LAB_LAB_CONCAT

#ifndef LAB_MEM_LIMIT_MB
#define LAB_MEM_LIMIT_MB 32
#endif

#define LAB_ARG(T, name)                               \
 public:                                               \
  inline void name(const T& new_##name) { /* NOLINT */ \
    this->name##_ = new_##name;                        \
  }                                                    \
  inline void name(T&& new_##name) { /* NOLINT */      \
    this->name##_ = std::move(new_##name);             \
  }                                                    \
  inline const T& name() const noexcept { /* NOLINT */ \
    return this->name##_;                              \
  }                                                    \
  inline T& name() noexcept { /* NOLINT */             \
    return this->name##_;                              \
  }                                                    \
                                                       \
 protected:                                            \
  T name##_ /* NOLINT */

#define LAB_NONCOPYABLE(class_name)       \
  class_name(const class_name&) = delete; \
  class_name& operator=(const class_name&) = delete
#define LAB_NONCOPYMOVEABLE(class_name)              \
  class_name(const class_name&) = delete;            \
  class_name& operator=(const class_name&) = delete; \
  class_name(class_name&&) = delete;                 \
  class_name& operator=(class_name&&) = delete
#define LAB_DEFAULT_CONSTRUCT(Name)                 \
  Name() = default;                                 \
  Name(const Name& other) = default;                \
  Name& operator=(const Name& other) = default;     \
  Name(Name&& other) noexcept = default;            \
  Name& operator=(Name&& other) noexcept = default; \
  virtual ~Name() = default

// Suppresses 'unused variable' warning

namespace lab::internal {
template <typename T>
LAB_FORCE_INLINE void ignore_unused_variable(const T&) {}

template <typename T, uint64_t N>
auto array_size_helper(const T (&array)[N]) -> char (&)[N];
} // namespace lab::internal

#define LAB_UNUSED_VARIABLE(var) ::lab::internal::ignore_unused_variable(var);
#define LAB_ARRAYSIZE(array) (sizeof(::lab::internal::array_size_helper(array)))

// Math constants

namespace lab::math {

constexpr double Epsilon = 0.0001;
constexpr double Pi = 3.14159265358979323846;
constexpr double InvPi = 0.31830988618379067154;
constexpr double Inv2Pi = 0.15915494309189533577;
constexpr double Inv4Pi = 0.07957747154594766788;
constexpr double PiOver2 = 1.57079632679489661923;
constexpr double PiOver4 = 0.78539816339744830961;
constexpr double Sqrt2 = 1.41421356237309504880;
constexpr float Infinity = std::numeric_limits<float>::infinity();
constexpr float DInfinity = std::numeric_limits<double>::infinity();
const double Min = std::numeric_limits<double>::min();
const double Max = std::numeric_limits<double>::max();

inline double safe_sqrt(double x) {
  assert(x >= -1e-3);
  return std::sqrt(std::max(0., x));
}

} // namespace lab::math

namespace lab::utils {
LAB_FORCE_INLINE bool is_torch_cuda_available() {
  return torch::cuda::is_available();
}

LAB_FORCE_INLINE torch::Device get_torch_device() {
  return is_torch_cuda_available() ? torch::kCUDA : torch::kCPU;
}
} // namespace lab::utils

#define LAB_TYPE_DECLARE(Name, Namespace)      \
  struct Name : public Namespace::Name {       \
    using Namespace::Name::Name;               \
    static constexpr const char* name = #Name; \
  }
