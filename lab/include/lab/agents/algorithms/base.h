#pragma once

#include "lab/agents/net/base.h"
#include "lab/agents/net/mlp.h"
#include "lab/utils/policy.h"

namespace lab
{
namespace agents
{

class Body;

class Algorithm
{
    LAB_ARG(std::shared_ptr<Body>, body);
    LAB_ARG(std::shared_ptr<NetImpl>, net);
    LAB_ARG(std::shared_ptr<torch::optim::Optimizer>, optimizer);
    LAB_ARG(std::shared_ptr<torch::optim::LRScheduler>, lrscheduler);
    LAB_ARG(utils::AlgorithmSpec, spec);
    LAB_ARG(utils::VarScheduler, explore_var_scheduler);
    LAB_ARG(utils::VarScheduler, entropy_coef_scheduler);
public:
    LAB_DEFAULT_CONSTRUCT(Algorithm);
    Algorithm(const std::shared_ptr<Body>& body, const utils::AlgorithmSpec& spec)
        : body_(body), spec_(std::move(spec))
    {
        init_algorithm_params();
        init_nets();
    }

    void init_algorithm_params()
    {
        LAB_UNIMPLEMENTED;
    }

    void init_nets()
    {
        LAB_UNIMPLEMENTED;
    }

    torch::Tensor train()
    {
        LAB_UNIMPLEMENTED;
        return torch::Tensor();
    }

    void update()
    {
        LAB_UNIMPLEMENTED;
    }

    torch::IValue act(const torch::Tensor& state)
    {
        LAB_UNIMPLEMENTED;
        return torch::IValue();
    }

    torch::Tensor sample()
    {
        LAB_UNIMPLEMENTED;
        return torch::Tensor();
    }

    void save(torch::serialize::OutputArchive& archive) const
    {

    }

    void load(torch::serialize::InputArchive& archive)
    {

    }
protected:
    std::shared_ptr<NetImpl> build_net(const utils::NetSpec& spec, int64_t in_dim, int64_t out_dim)
    {
        if(spec.type == utils::NetType::MLPNET)
            return std::make_shared<MLPNetImpl>(std::move(spec), in_dim, out_dim);

        LAB_LOG_FATAL("Unsupported net!");
        return nullptr;
    }
};

LAB_FORCE_INLINE torch::serialize::OutputArchive& operator<<(torch::serialize::OutputArchive& archive, const std::shared_ptr<Algorithm>& algo)
{
    return archive;
}

LAB_FORCE_INLINE torch::serialize::InputArchive& operator>>(torch::serialize::InputArchive& archive, const std::shared_ptr<Algorithm>& algo)
{
    return archive;
}

}
}