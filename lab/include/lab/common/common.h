#pragma once

#include <iostream>
#include <memory>
#include <utility>
#include <algorithm>
#include <functional>
#include <string>
#include <sstream>
#include <array>
#include <vector>
#include <unordered_map>
#include <map>
#include <unordered_set>
#include <set>
#include <cstdint>
#include <fstream>
#include <type_traits>
#include <cstddef>
#include <cassert>
#include <new>
#include <thread>
#include <deque>
#include <list>
#include <inttypes.h>
#include <codecvt>
#include <locale>
#include <ctype.h>
#include <mutex>
#include <queue>
#include <random>
#include <iomanip>
#include <chrono>
#include <filesystem>
#include <atomic>
#include <condition_variable>
#include <shared_mutex>
#include <optional>
#include <regex>

#include <torch/torch.h>

#include "lab/common/version.h"
#include "lab/common/forwards.h"
#include "lab/common/platformdetect.h"

#ifdef LAB_DEBUG
#ifdef LAB_PLATFORM_WINDOWS
#define LAB_BREAK() __debugbreak()
#else
#include <signal.h>
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


#ifdef LAB_ENABLE_LOG
#include "lab/common/logger.h"
#define LAB_INIT_LOG() ::lab::common::Logger::init()
#define LAB_SHUTDOWN_LOG() ::lab::common::Logger::shutdown()

#define LAB_LOG_TRACE_(...) SPDLOG_LOGGER_CALL(::lab::common::Logger::get_core_logger(), spdlog::level::level_enum::trace, __VA_ARGS__)
#define LAB_LOG_DEBUG_(...) SPDLOG_LOGGER_CALL(::lab::common::Logger::get_core_logger(), spdlog::level::level_enum::debug, __VA_ARGS__)
#define LAB_LOG_INFO_(...)  SPDLOG_LOGGER_CALL(::lab::common::Logger::get_core_logger(), spdlog::level::level_enum::info, __VA_ARGS__)
#define LAB_LOG_WARN_(...)  SPDLOG_LOGGER_CALL(::lab::common::Logger::get_core_logger(), spdlog::level::level_enum::warn, __VA_ARGS__)
#define LAB_LOG_ERROR_(...) SPDLOG_LOGGER_CALL(::lab::common::Logger::get_core_logger(), spdlog::level::level_enum::err, __VA_ARGS__)
#define LAB_LOG_FATAL_(...) SPDLOG_LOGGER_CALL(::lab::common::Logger::get_core_logger(), spdlog::level::level_enum::critical, __VA_ARGS__), LAB_BREAK()

#define LAB_LOG_TRACE(msg, ...) LAB_LOG_TRACE_(msg "\n" __VA_OPT__(,) __VA_ARGS__)
#define LAB_LOG_DEBUG(msg, ...) LAB_LOG_DEBUG_(msg "\n" __VA_OPT__(,) __VA_ARGS__)
#define LAB_LOG_INFO(msg, ...)  LAB_LOG_INFO_(msg "\n" __VA_OPT__(,) __VA_ARGS__)
#define LAB_LOG_WARN(msg, ...)  LAB_LOG_WARN_(msg "\n" __VA_OPT__(,) __VA_ARGS__)
#define LAB_LOG_ERROR(msg, ...) LAB_LOG_ERROR_(msg "\n" __VA_OPT__(,) __VA_ARGS__)
#define LAB_LOG_FATAL(msg, ...) LAB_LOG_FATAL_(msg "\n" __VA_OPT__(,) __VA_ARGS__)

#else
#define LAB_INIT_LOG() ((void)0)
#define LAB_SHUTDOWN_LOG() ((void)0)

#define LAB_LOG_TRACE(...) ((void)0)
#define LAB_LOG_DEBUG(...) ((void)0)
#define LAB_LOG_INFO(...) ((void)0)
#define LAB_LOG_WARN(...) ((void)0)
#define LAB_LOG_ERROR(...) ((void)0)
#define LAB_LOG_FATAL(...) ((void)0)
#endif

#define LAB_UNIMPLEMENTED LAB_LOG_FATAL("Unimplemented: {} : {}", __FILE__, __LINE__)
#define LAB_UNREACHABLE LAB_LOG_FATAL("Unexpected: {} : {}", __FILE__, __LINE__)

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
#define DISABLE_WARNING(warningNumber) __pragma(warning(disable \
                                                : warningNumber))
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
  // NOTE MSVC often gives C4127 warnings with compiletime if statements. See bug 1362.
  // This workaround is ugly, but it does the job.
#define LAB_CONST_CONDITIONAL(cond)  (void)0, cond
#else
#define LAB_CONST_CONDITIONAL(cond)  cond
#endif

// Check
#define EMPTY_CHECK \
	do {			\
	} while (false);
#define CHECK_FAIL_RAISE(x) LAB_LOG_FATAL("Check failed: {} at {}, line {}", x, __FILE__, __LINE__), true
#define CHECK_(x) (!(!(x) && (CHECK_FAIL_RAISE(#x))))
#define CHECK_MSG(x, msg) (!(!(x) && (LAB_LOG_ERROR(#msg), true)))
#define CHECK_IMPL(a, b, op) (!(!(a op b) && (CHECK_FAIL_RAISE(#a #op #b))))
#define CHECK_EQ(a, b) CHECK_IMPL(a, b, ==)
#define CHECK_NE(a, b) CHECK_IMPL(a, b, !=)
#define CHECK_GT(a, b) CHECK_IMPL(a, b, >)
#define CHECK_GE(a, b) CHECK_IMPL(a, b, >=)
#define CHECK_LT(a, b) CHECK_IMPL(a, b, <)
#define CHECK_LE(a, b) CHECK_IMPL(a, b, <=)

#ifdef LAB_ENABLE_CHECK
#define LAB_CHECK(x) (CHECK_(x))
#define LAB_CHECK_MSG(x, msg) (CHECK_MSG(x, msg))
#define LAB_CHECK_EQ(a, b) (CHECK_EQ(a, b))
#define LAB_CHECK_NE(a, b) (CHECK_NE(a, b))
#define LAB_CHECK_GT(a, b) (CHECK_GT(a, b))
#define LAB_CHECK_GE(a, b) (CHECK_GE(a, b))
#define LAB_CHECK_LT(a, b) (CHECK_LT(a, b))
#define LAB_CHECK_LE(a, b) (CHECK_LE(a, b))
#else
#define LAB_CHECK(x) EMPTY_CHECK
#define LAB_CHECK_MSG(x, msg) EMPTY_CHECK
#define LAB_CHECK_MSG_RET(x, ret, msg) EMPTY_CHECK
#define LAB_CHECK_EQ(a, b) EMPTY_CHECK
#define LAB_CHECK_NE(a, b) EMPTY_CHECK
#define LAB_CHECK_GT(a, b) EMPTY_CHECK
#define LAB_CHECK_GE(a, b) EMPTY_CHECK
#define LAB_CHECK_LT(a, b) EMPTY_CHECK
#define LAB_CHECK_LE(a, b) EMPTY_CHECK
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

#define LAB_ARG(T, name)                                                  \
 public:                                                                  \
  inline void name(const T& new_##name) { /* NOLINT */                    \
    this->name##_ = new_##name;                                           \
  }                                                                       \
  inline void name(T&& new_##name) { /* NOLINT */                         \
    this->name##_ = std::move(new_##name);                                \
  }                                                                       \
  inline const T& name() const noexcept { /* NOLINT */                    \
    return this->name##_;                                                 \
  }                                                                       \
  inline T& name() noexcept { /* NOLINT */                                \
    return this->name##_;                                                 \
  }                                                                       \
                                                                          \
 protected:                                                                 \
  T name##_ /* NOLINT */


#define LAB_NONCOPYABLE(class_name)							  \
    class_name(const class_name&)            = delete;		  \
    class_name& operator=(const class_name&) = delete
#define LAB_NONCOPYMOVEABLE(class_name)					  \
    class_name(const class_name&)            = delete;		  \
    class_name& operator=(const class_name&) = delete;		  \
    class_name(class_name&&)                 = delete;		  \
    class_name& operator=(class_name&&)      = delete
#define LAB_DEFAULT_CONSTRUCT(Name)                       \
    Name() = default;                                     \
    Name(const Name& other) = default;                    \
    Name& operator=(const Name& other) = default;         \
    Name(Name&& other) noexcept = default;                \
    Name& operator=(Name&& other) noexcept = default;     \
    virtual ~Name() = default

// Suppresses 'unused variable' warning
namespace lab
{
    namespace internal
    {
        template<typename T>
        LAB_FORCE_INLINE void ignore_unused_variable(const T&) {}

        template <typename T, uint64_t N>
        auto array_size_helper(const T(&array)[N]) -> char(&)[N];
    }
}
#define LAB_UNUSED_VARIABLE(var) ::lab::internal::ignore_unused_variable(var);
#define LAB_ARRAYSIZE(array) (sizeof(::lab::internal::array_size_helper(array)))

// Math constants
namespace lab
{
namespace math
{

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

inline double safe_sqrt(double x) 
{
  assert(x >= -1e-3);
  return std::sqrt(std::max(0., x));
}

}
}

namespace lab
{
namespace utils
{
LAB_FORCE_INLINE bool is_torch_cuda_available()
{
    return torch::cuda::is_available();
}

LAB_FORCE_INLINE torch::Device get_torch_device()
{
    return is_torch_cuda_available() ? torch::kCUDA : torch::kCPU;
}
}
}

#define LAB_TYPE_DECLARE(Name, Namespace)                                 \
struct Name : public Namespace::Name                                      \
{                                                                         \
    using Namespace::Name::Name;                                          \
    static constexpr const char* name = #Name;                            \
}