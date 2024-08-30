#pragma once

#include <common/common.h>
#include <torch/torch.h>
#include "lab/version.h"

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
