#pragma once

#include <common/common.h>

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

#include "renderer/version.h"