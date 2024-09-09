#pragma once

#include "lab/common/common.h"
#include "lab/utils/net.h"
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
    LAB_ARG(std::shared_ptr<lab::utils::ActivationModule>, hid_layers_activation);
    LAB_ARG(std::vector<std::shared_ptr<lab::utils::ActivationModule>>, out_layers_activations);
    LAB_ARG(std::shared_ptr<lab::utils::LossModule>, loss_function);
    LAB_ARG(torch::Device, device) = torch::kCPU;
public:
    using torch::nn::Module::Module;
    NetImpl(const utils::NetSpec& spec, int64_t in_dim, torch::IntArrayRef out_dim);

    virtual torch::Tensor forward(torch::Tensor x) = 0;

    torch::Tensor& train_step(
        torch::Tensor& loss, 
        const std::shared_ptr<torch::optim::Optimizer>& optimizer, 
        const std::shared_ptr<torch::optim::LRScheduler>& lr_scheduler,
        const std::shared_ptr<utils::Clock>& clock);

    // debug only
    void print_weights() const;
};

TORCH_MODULE(Net);

}
}