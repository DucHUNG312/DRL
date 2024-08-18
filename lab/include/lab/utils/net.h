#pragma once

#include "lab/core.h"
#include "lab/utils/optimizer.h"
#include "lab/utils/spec.h"

namespace lab
{

namespace utils
{

torch::nn::Sequential build_fc_model(const std::vector<int64_t>& dims, const torch::nn::AnyModule& activation);

torch::nn::AnyModule get_act_fn(const std::string& name);

torch::nn::AnyModule get_loss_fn(const std::string& name);

std::shared_ptr<torch::optim::Optimizer> get_optim(const torch::nn::Module& net, const OptimSpec& optim_spec);

std::shared_ptr<torch::optim::LRScheduler> get_lr_schedular(torch::optim::Optimizer& optimizer, const LrSchedulerSpec& spec);

torch::nn::init::NonlinearityType get_nonlinearirty_type(const std::string& name);

}

}
