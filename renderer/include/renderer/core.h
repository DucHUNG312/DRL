#pragma once

#include <common/common.h>

#include "renderer/version.h"

// GPU Macro Definitions
#ifdef LAB_GPU_BUILD
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#ifndef LAB_NOINLINE
#define LAB_NOINLINE __attribute__((noinline))
#endif
#define LAB_GLOBAL __global__
#define LAB_CPU_GPU __host__ __device__
#define LAB_GPU __device__
#define LAB_CONST __device__ const

#else
#define LAB_GLOBAL 
#define LAB_CPU_GPU
#define LAB_GPU
#define LAB_CONST const
#endif

namespace lab
{

namespace math
{

template <typename T>
class Vector2;
template <typename T>
class Vector3;
template <typename T>
class Point3;
template <typename T>
class Point2;
template <typename T>
class Normal3;
using Point2f = Point2<float>;
using Point2i = Point2<int>;
using Point3f = Point3<float>;
using Vector2f = Vector2<float>;
using Vector2i = Vector2<int>;
using Vector3f = Vector3<float>;

}   

}

