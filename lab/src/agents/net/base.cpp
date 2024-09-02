#include "lab/agents/net/base.h"
#include "lab/utils/clock.h"

namespace lab
{

namespace agents
{

NetImpl::NetImpl(const utils::NetSpec& spec, int64_t in_dim, torch::IntArrayRef out_dim)
    : spec_(std::move(spec)), in_dim_(in_dim), out_dim_(out_dim)
{
    if(spec_.gpu && utils::get_torch_device() == torch::kCUDA)
        device_ = torch::kCUDA;
}

void NetImpl::train_step(
    torch::Tensor& loss, 
    const std::shared_ptr<torch::optim::Optimizer>& optimizer, 
    const std::shared_ptr<torch::optim::LRScheduler>& lr_scheduler,
    const std::shared_ptr<utils::Clock>& clock)
{
    loss.to(device_);
    lr_scheduler->step();
    optimizer->zero_grad();
    loss.backward();
    optimizer->step();
    clock->tick_opt_step();
}

void NetImpl::print_weights() const
{
    for (const auto& param : parameters()) 
        if (param.grad().defined())
            LAB_LOG_DEBUG("Gradient norm: {}", param.grad().norm().item<double>());
}

}

}