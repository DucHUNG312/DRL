#include "lab/agents/net/base.h"

namespace lab
{

namespace agents
{

Net::Net(const utils::NetSpec& spec, int64_t in_dim, torch::IntArrayRef out_dim)
{
    spec_ = spec;
    in_dim_ = in_dim;
    out_dim_ = out_dim;

    if(spec_.gpu && utils::get_torch_device() == torch::kCUDA)
        device_ = torch::kCUDA;
}

torch::Tensor Net::forward(torch::Tensor x)
{
    LAB_UNIMPLEMENTED;
    return torch::Tensor();
}

torch::Tensor Net::train_step(
    torch::Tensor loss, 
    torch::optim::Optimizer& optimizer, 
    torch::optim::LRScheduler& lr_scheduler,
    utils::Clock& clock)
{
    lr_scheduler.step();
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
    clock.tick_opt_step();
    return loss;
}

}

}