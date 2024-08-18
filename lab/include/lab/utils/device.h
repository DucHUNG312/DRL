#pragma once

#include "lab/core.h"

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