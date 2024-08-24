#pragma once

#include "lab/agents/base.h"

namespace lab
{
namespace agents
{

class Net
{
    LAB_ARG(utils::NetSpec, spec);
    LAB_ARG(int64_t, in_dim);
    LAB_ARG(torch::IntArrayRef, out_dim);
    LAB_ARG(torch::Device, device) = torch::kCPU;
public:
    Net(const utils::NetSpec& spec, int64_t in_dim, torch::IntArrayRef out_dim);
    LAB_DEFAULT_CONSTRUCT(Net);

    torch::Tensor forward(torch::Tensor x);

    torch::Tensor train_step(
        torch::Tensor loss, 
        torch::optim::Optimizer& optimizer, 
        torch::optim::LRScheduler& lr_scheduler,
        utils::Clock& clock);
};

}
}