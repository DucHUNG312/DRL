#pragma once

#include "lab/agents/memory/base.h"
#include "lab/envs/base.h"

namespace lab
{

namespace agents
{

class Reinforce : public Algorithm
{
public:
    using Algorithm::Algorithm;

    void init_algorithm_params()
    {
        LAB_UNIMPLEMENTED;
    }

    void init_nets()
    {
        int64_t in_dim = body_->env()->get_state_dim();
        int64_t out_dim = body_->env()->get_action_dim();
        net_ = build_net(body_->spec().net, in_dim, out_dim);
        optimizer_ = utils::get_optim(net_, net_->spec().optim_spec);
        lrscheduler_ = utils::get_lr_schedular(optimizer_, net_->spec().lr_scheduler_spec);
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
private:
    bool to_train = false;
};

}

}