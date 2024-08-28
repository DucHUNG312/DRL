#pragma once

#include "lab/core.h"
#include "lab/utils/env.h"
#include "lab/utils/spec.h"
#include "lab/utils/net.h"

namespace lab
{
namespace agents
{

class NetImpl : public torch::nn::Module
{
    LAB_ARG(utils::NetSpec, spec);
    LAB_ARG(int64_t, in_dim);
    LAB_ARG(torch::IntArrayRef, out_dim);
    LAB_ARG(std::shared_ptr<lab::utils::Module>, hid_layers_activation);
    LAB_ARG(std::vector<std::shared_ptr<lab::utils::Module>>, out_layers_activations);
    LAB_ARG(std::shared_ptr<lab::utils::Module>, loss_function);
    LAB_ARG(torch::Device, device) = torch::kCPU;
public:
    using torch::nn::Module::Module;
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