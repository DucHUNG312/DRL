#pragma once

#include "lab/envs/base.h"
#include "lab/agents/net/base.h"
#include "lab/agents/net/mlp.h"
#include "lab/utils/policy.h"
#include "lab/utils/optimizer.h"

namespace lab
{
namespace agents
{

class Algorithm : public std::enable_shared_from_this<Algorithm>
{
    LAB_ARG(std::shared_ptr<envs::Env>, env);
    LAB_ARG(std::shared_ptr<lab::agents::NetImpl>, net);
    LAB_ARG(std::shared_ptr<torch::optim::Optimizer>, optimizer);
    LAB_ARG(std::shared_ptr<torch::optim::LRScheduler>, lrscheduler);
    LAB_ARG(utils::AlgorithmSpec, spec);
    LAB_ARG(utils::ActionPolicy, policy);
    LAB_ARG(utils::VarScheduler, explore_var_scheduler);
    LAB_ARG(utils::VarScheduler, entropy_coef_scheduler);
    LAB_ARG(torch::Tensor, loss);
    LAB_ARG(double, explore_var);
    LAB_ARG(double, entropy_coef);
    LAB_ARG(double, learning_rate);
    LAB_ARG(bool, to_train) = false;
    LAB_ARG(bool, center_return) = false;
public:
public:
    using ExperienceDict = torch::Dict<std::string, torch::List<torch::IValue>>;

    Algorithm(const utils::AlgorithmSpec& spec);

    virtual torch::Tensor train(const ExperienceDict& experiences) = 0;

    virtual void update(const torch::Tensor& loss) = 0;

    virtual torch::Tensor act(const torch::Tensor& state) = 0;

    virtual torch::Tensor calc_pdparam(const torch::Tensor& x);

    virtual torch::Tensor calc_pdparam_batch(const ExperienceDict& experiences);

    void save(torch::serialize::OutputArchive& archive) const;

    void load(torch::serialize::InputArchive& archive);
};

torch::serialize::OutputArchive& operator<<(torch::serialize::OutputArchive& archive, const std::shared_ptr<Algorithm>& algo);

torch::serialize::InputArchive& operator>>(torch::serialize::InputArchive& archive, const std::shared_ptr<Algorithm>& algo);

}
}