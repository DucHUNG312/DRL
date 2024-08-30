#pragma once

#include "lab/agents/net/base.h"
#include "lab/agents/net/mlp.h"
#include "lab/utils/policy.h"
#include "lab/utils/optimizer.h"

namespace lab
{
namespace agents
{

class Body;

class Algorithm
{
    LAB_ARG(std::shared_ptr<Body>, body);
    LAB_ARG(std::shared_ptr<lab::agents::NetImpl>, net);
    LAB_ARG(std::shared_ptr<torch::optim::Optimizer>, optimizer);
    LAB_ARG(std::shared_ptr<torch::optim::LRScheduler>, lrscheduler);
    LAB_ARG(utils::AlgorithmSpec, spec);
    LAB_ARG(utils::ActionPolicy, policy);
    LAB_ARG(utils::VarScheduler, explore_var_scheduler);
    LAB_ARG(utils::VarScheduler, entropy_coef_scheduler);
public:
    Algorithm(const std::shared_ptr<Body>& body, const utils::AlgorithmSpec& spec);

    torch::Tensor train();

    void update();

    torch::IValue act(const torch::Tensor& state);

    torch::Tensor sample();

    torch::Tensor calc_pdparam(torch::Tensor x);

    void save(torch::serialize::OutputArchive& archive) const;

    void load(torch::serialize::InputArchive& archive);
};

torch::serialize::OutputArchive& operator<<(torch::serialize::OutputArchive& archive, const std::shared_ptr<Algorithm>& algo);

torch::serialize::InputArchive& operator>>(torch::serialize::InputArchive& archive, const std::shared_ptr<Algorithm>& algo);

}
}