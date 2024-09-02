#pragma once

#include "renderer/core.h"

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
namespace render
{
namespace cuda
{

LAB_FORCE_INLINE void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
    if (result != cudaSuccess)
    {
        LAB_LOG_FATAL("CUDA Runtime API Error({}) {}, at {}: {}({})", result, cudaGetErrorString(result), func, file, line);
        cudaDeviceReset();
        exit(99);
    }
}

LAB_FORCE_INLINE void check_cuda_driver(CUresult result, char const* const func, const char* const file, int const line)
{
    if (result != CUDA_SUCCESS) 
    {
        const char *errorName, *errorStr;
        cuGetErrorName(result, &errorName);
        cuGetErrorString(result, &errorStr);
        LAB_LOG_FATAL("CUDA Driver API Error({}) {}-{} at {}: {}({})", result, errorName, errorStr, func, file, line);
        cudaDeviceReset();
        exit(99);
    }
}

#define LAB_CHECK_CUDA(val) ::lab::render::cuda::check_cuda( (val), #val, __FILE__, __LINE__ )
#define LAB_CHECK_CUDA_DRIVER(val) ::lab::render::cuda::check_cuda_driver( (val), #val, __FILE__, __LINE__ )

LAB_FORCE_INLINE void* alloc_gpu(size_t num_bytes)
{
    void* ptr;
    LAB_CHECK_CUDA(cudaMalloc(&ptr, num_bytes));
    return ptr;
}

LAB_FORCE_INLINE void dealloc_gpu(void *ptr)
{
    LAB_CHECK_CUDA(cudaFree(ptr));
}

LAB_FORCE_INLINE void* alloc_staging(size_t num_bytes)
{
    void* ptr;
    LAB_CHECK_CUDA(cudaHostAlloc(&ptr, num_bytes, cudaHostAllocMapped | cudaHostAllocWriteCombined));
    return ptr;
}

LAB_FORCE_INLINE void* alloc_readback(size_t num_bytes)
{
    void* ptr;
    LAB_CHECK_CUDA(cudaHostAlloc(&ptr, num_bytes, cudaHostAllocMapped));
    return ptr;
}

LAB_FORCE_INLINE void dealloc_cpu(void *ptr)
{
    LAB_CHECK_CUDA(cudaFreeHost(ptr));
}

LAB_FORCE_INLINE void copy_cpu_to_gpu(cudaStream_t strm, void *gpu, void *cpu, size_t num_bytes)
{
    LAB_CHECK_CUDA(cudaMemcpyAsync(gpu, cpu, num_bytes, cudaMemcpyHostToDevice, strm));
}

LAB_FORCE_INLINE void copy_gpu_to_cpu(cudaStream_t strm, void *cpu, void *gpu, size_t num_bytes)
{
    LAB_CHECK_CUDA(cudaMemcpyAsync(cpu, gpu, num_bytes, cudaMemcpyDeviceToHost, strm));
}

LAB_FORCE_INLINE cudaStream_t make_stream()
{
    cudaStream_t strm;
    LAB_CHECK_CUDA(cudaStreamCreate(&strm));
    return strm;
}

LAB_FORCE_INLINE int cuda_runtime_version() 
{
	int version;
	LAB_CHECK_CUDA(cudaRuntimeGetVersion(&version));
	return version;
}

LAB_FORCE_INLINE int cuda_device() 
{
	int device;
	LAB_CHECK_CUDA(cudaGetDevice(&device));
	return device;
}

LAB_FORCE_INLINE void set_cuda_device(int device) 
{
	LAB_CHECK_CUDA(cudaSetDevice(device));
}

LAB_FORCE_INLINE int cuda_device_count() 
{
	int device_count;
	LAB_CHECK_CUDA(cudaGetDeviceCount(&device_count));
	return device_count;
}

LAB_FORCE_INLINE bool cuda_supports_virtual_memory(int device) 
{
	int supports_vmm;
	LAB_CHECK_CUDA_DRIVER(cuDeviceGetAttribute(&supports_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, device));
	return supports_vmm != 0;
}

LAB_FORCE_INLINE std::string cuda_device_name(int device) 
{
	cudaDeviceProp props;
	LAB_CHECK_CUDA(cudaGetDeviceProperties(&props, device));
	return props.name;
}

LAB_FORCE_INLINE uint32_t cuda_compute_capability(int device) 
{
	cudaDeviceProp props;
	LAB_CHECK_CUDA(cudaGetDeviceProperties(&props, device));
	return props.major * 10 + props.minor;
}

LAB_FORCE_INLINE uint32_t cuda_max_supported_compute_capability() 
{
	int cuda_version = cuda_runtime_version();
	if (cuda_version < 11000) return 75;
	else if (cuda_version < 11010) return 80;
	else if (cuda_version < 11080) return 86;
	else return 90;
}

LAB_FORCE_INLINE uint32_t cuda_supported_compute_capability(int device) 
{
	return std::min(cuda_compute_capability(device), cuda_max_supported_compute_capability());
}

}
}
}