#pragma once

#include "lab/core.h"
#include "lab/utils/env.h"
#include "lab/utils/spec.h"

namespace lab
{
namespace agents
{

class NetImpl : public torch::nn::Module
{
    LAB_ARG(utils::NetSpec, spec);
    LAB_ARG(int64_t, in_dim);
    LAB_ARG(torch::IntArrayRef, out_dim);
    LAB_ARG(torch::Device, device) = torch::kCPU;
public:
    NetImpl(const utils::NetSpec& spec, int64_t in_dim, torch::IntArrayRef out_dim);

    torch::Tensor forward(torch::Tensor x);

    torch::Tensor train_step(
        torch::Tensor loss, 
        torch::optim::Optimizer& optimizer, 
        torch::optim::LRScheduler& lr_scheduler,
        utils::Clock& clock);
};

TORCH_MODULE(Net);

}
}