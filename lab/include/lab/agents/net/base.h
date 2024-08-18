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
    Net() = default;
    Net(const utils::NetSpec& spec, int64_t in_dim, torch::IntArrayRef out_dim)
    {
        spec_ = spec;
        in_dim_ = in_dim;
        out_dim_ = out_dim;

        if(spec_.gpu && utils::get_torch_device() == torch::kCUDA)
            device_ = torch::kCUDA;
    }

    virtual torch::Tensor forward(torch::Tensor x)
    {
        LAB_UNIMPLEMENTED;
        return torch::Tensor();
    }

    virtual torch::Tensor train_step(
        torch::Tensor loss, 
        torch::optim::Optimizer& optimizer, 
        torch::optim::LRScheduler& lr_scheduler,
        utils::Clock& clock
        )
    {
        lr_scheduler.step();
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        clock.tick_opt_step();
        return loss;
    }
};

}
}