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
#include <opencv2/opencv.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

namespace lab
{
// double vectors
using Vec2  = Eigen::Matrix<double, 2, 1>;                          
using Vec3  = Eigen::Matrix<double, 3, 1>;                          
using Vec4  = Eigen::Matrix<double, 4, 1>;                          
using Vec5  = Eigen::Matrix<double, 5, 1>;                          
using Vec6  = Eigen::Matrix<double, 6, 1>;                          
using Vec7  = Eigen::Matrix<double, 7, 1>;                          
using Vec8  = Eigen::Matrix<double, 8, 1>;                          
using Vec9  = Eigen::Matrix<double, 9, 1>;                          
using Vec10 = Eigen::Matrix<double, 10, 1>;                         
using Vec   = Eigen::Matrix<double, Eigen::Dynamic, 1>;             

// float square matrixs
using Mat2  = Eigen::Matrix<double, 2, 2>;                          
using Mat3  = Eigen::Matrix<double, 3, 3>;                          
using Mat4  = Eigen::Matrix<double, 4, 4>;                          
using Mat5  = Eigen::Matrix<double, 5, 5>;                          
using Mat6  = Eigen::Matrix<double, 6, 6>;                          
using Mat7  = Eigen::Matrix<double, 7, 7>;                          
using Mat8  = Eigen::Matrix<double, 8, 8>;                          
using Mat9  = Eigen::Matrix<double, 9, 9>;                          
using Mat10 = Eigen::Matrix<double, 10, 10>;                        
using Mat   = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

// float vectors
using Vec2f  = Eigen::Matrix<float, 2, 1>;                          
using Vec3f  = Eigen::Matrix<float, 3, 1>;                          
using Vec4f  = Eigen::Matrix<float, 4, 1>;                          
using Vec5f  = Eigen::Matrix<float, 5, 1>;                          
using Vec6f  = Eigen::Matrix<float, 6, 1>;                          
using Vec7f  = Eigen::Matrix<float, 7, 1>;                          
using Vec8f  = Eigen::Matrix<float, 8, 1>;                          
using Vec9f  = Eigen::Matrix<float, 9, 1>;                          
using Vec10f = Eigen::Matrix<float, 10, 1>;                         
using Vecf   = Eigen::Matrix<float, Eigen::Dynamic, 1>;             

// float square matrixs
using Mat2f  = Eigen::Matrix<float, 2, 2>;                          
using Mat3f  = Eigen::Matrix<float, 3, 3>;                          
using Mat4f  = Eigen::Matrix<float, 4, 4>;                          
using Mat5f  = Eigen::Matrix<float, 5, 5>;                          
using Mat6f  = Eigen::Matrix<float, 6, 6>;                          
using Mat7f  = Eigen::Matrix<float, 7, 7>;                          
using Mat8f  = Eigen::Matrix<float, 8, 8>;                          
using Mat9f  = Eigen::Matrix<float, 9, 9>;                          
using Mat10f = Eigen::Matrix<float, 10, 10>;                        
using Matf   = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
}

#include "version.h"

#if defined(_WIN32) || defined(_WIN64)
    #define LAB_PLATFORM_WINDOWS
#elif defined(__APPLE__) || defined(__MACH__)
    #define LAB_PLATFORM_MACOS
#elif defined(__linux__)
    #define LAB_PLATFORM_LINUX
#else
    #define LAB_PLATFORM_UNKNOWN
#endif
#ifdef LAB_PLATFORM_UNKNOWN
    #error "Unknown platform. Compilation halted."
#endif

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
#endif // LAB_DEBUG
#ifdef LAB_ENABLE_LOG
#include "lab/logger.h"
#define LAB_INIT_LOG() ::lab::Logger::init()
#define LAB_SHUTDOWN_LOG() ::lab::Logger::shutdown()

#define LAB_LOG_TRACE_(...) SPDLOG_LOGGER_CALL(::lab::Logger::get_core_logger(), spdlog::level::level_enum::trace, __VA_ARGS__)
#define LAB_LOG_DEBUG_(...) SPDLOG_LOGGER_CALL(::lab::Logger::get_core_logger(), spdlog::level::level_enum::debug, __VA_ARGS__)
#define LAB_LOG_INFO_(...) SPDLOG_LOGGER_CALL(::lab::Logger::get_core_logger(), spdlog::level::level_enum::info, __VA_ARGS__)
#define LAB_LOG_WARN_(...) SPDLOG_LOGGER_CALL(::lab::Logger::get_core_logger(), spdlog::level::level_enum::warn, __VA_ARGS__)
#define LAB_LOG_ERROR_(...) SPDLOG_LOGGER_CALL(::lab::Logger::get_core_logger(), spdlog::level::level_enum::err, __VA_ARGS__)
#define LAB_LOG_FATAL_(...) SPDLOG_LOGGER_CALL(::lab::Logger::get_core_logger(), spdlog::level::level_enum::critical, __VA_ARGS__), LAB_BREAK()

#define LAB_LOG_TRACE(...) (SPDLOG_LOGGER_CALL(::lab::Logger::get_core_logger(), spdlog::level::level_enum::trace, __VA_ARGS__), LAB_LOG_TRACE_('\n'))
#define LAB_LOG_DEBUG(...) (SPDLOG_LOGGER_CALL(::lab::Logger::get_core_logger(), spdlog::level::level_enum::debug, __VA_ARGS__), LAB_LOG_DEBUG_('\n'))
#define LAB_LOG_INFO(...) (SPDLOG_LOGGER_CALL(::lab::Logger::get_core_logger(), spdlog::level::level_enum::info, __VA_ARGS__), LAB_LOG_INFO_('\n'))
#define LAB_LOG_WARN(...) (SPDLOG_LOGGER_CALL(::lab::Logger::get_core_logger(), spdlog::level::level_enum::warn, __VA_ARGS__), LAB_LOG_WARN_('\n'))
#define LAB_LOG_ERROR(...) (SPDLOG_LOGGER_CALL(::lab::Logger::get_core_logger(), spdlog::level::level_enum::err, __VA_ARGS__), LAB_LOG_ERROR_('\n'))
#define LAB_LOG_FATAL(...) (SPDLOG_LOGGER_CALL(::lab::Logger::get_core_logger(), spdlog::level::level_enum::critical, __VA_ARGS__), LAB_LOG_FATAL_('\n'))
#else
#define LAB_INIT_LOG() ((void)0)
#define LAB_SHUTDOWN_LOG() ((void)0)

#define LAB_LOG_TRACE(...) ((void)0)
#define LAB_LOG_DEBUG(...) ((void)0)
#define LAB_LOG_INFO(...) ((void)0)
#define LAB_LOG_WARN(...) ((void)0)
#define LAB_LOG_ERROR(...) ((void)0)
#define LAB_LOG_FATAL(...) ((void)0)

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

// enable check
#define LAB_ENABLE_CHECK
#ifdef LAB_ENABLE_CHECK
#define CHECK_FAIL_RAISE(x) LAB_LOG_FATAL("Check failed: {} at {}, line {}", x, __FILE__, __LINE__), true
#define CHECK_(x) (!(!(x) && (CHECK_FAIL_RAISE(#x))))
#define CHECK_MSG(x, msg) (!(!(x) && (LAB_LOG_ERROR(#msg), true)))
#define CHECK_IMPL(a, b, op) (!(!(a op b) && (CHECK_FAIL_RAISE(#a #op #b))))
#define EMPTY_CHECK \
	do {			\
	} while (false);
#define CHECK_EQ(a, b) CHECK_IMPL(a, b, ==)
#define CHECK_NE(a, b) CHECK_IMPL(a, b, !=)
#define CHECK_GT(a, b) CHECK_IMPL(a, b, >)
#define CHECK_GE(a, b) CHECK_IMPL(a, b, >=)
#define CHECK_LT(a, b) CHECK_IMPL(a, b, <)
#define CHECK_LE(a, b) CHECK_IMPL(a, b, <=)

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
    class_name& operator=(const class_name&) = delete;
#define LAB_NONCOPYABLEANDMOVE(class_name)					  \
    class_name(const class_name&)            = delete;		  \
    class_name& operator=(const class_name&) = delete;		  \
    class_name(class_name&&)                 = delete;		  \
    class_name& operator=(class_name&&)      = delete;

#define LAB_CANCOPYABLE(class_name)							  \
    class_name(const class_name&)            = default;		  \
    class_name& operator=(const class_name&) = default;
#define LAB_CANCOPYABLEANDMOVE(class_name)					  \
    class_name(const class_name&)            = default;		  \
    class_name& operator=(const class_name&) = default;		  \
    class_name(class_name&&)                 = default;		  \
    class_name& operator=(class_name&&)      = default;

// Suppresses 'unused variable' warnings.
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

// GPU Macro Definitions
#if defined(LAB_GPU_BUILD)
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#ifndef LAB_NOINLINE
#define LAB_NOINLINE __attribute__((noinline))
#endif
#define LAB_CPU_GPU __host__ __device__
#define LAB_GPU __device__
#define LAB_CONST __device__ const

namespace lab
{
namespace cuda
{
    LAB_FORCE_INLINE bool is_cuda_available()
    {
        return torch::cuda::is_available();
    }

    LAB_FORCE_INLINE torch::Device get_device()
    {
        return is_cuda_available() ? torch::kCUDA : torch::kCPU;
    }
}
}

#define LAB_TORCH_DEVICE ::lab::cuda::get_device()
#else
#define LAB_CONST const
#define LAB_CPU_GPU
#define LAB_GPU
#define LAB_TORCH_DEVICE torch::kCPU
#endif

