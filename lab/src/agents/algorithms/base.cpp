#include "lab/agents/algorithms/base.h"
#include "lab/agents/body.h"
#include "lab/utils/net.h"

namespace lab
{

namespace agents
{

Algorithm::Algorithm(const std::shared_ptr<Body>& body, const utils::AlgorithmSpec& spec)
    : body_(body), spec_(std::move(spec))
{
    int64_t in_dim = body_->env()->get_state_dim();
    int64_t out_dim = body_->env()->get_action_dim();
    net_ = utils::create_net(body_->spec().net, in_dim, out_dim);
    optimizer_ = utils::create_optim(net_->spec().optim_spec.name, net_->parameters());
    lrscheduler_ = utils::create_lr_schedular(optimizer_, net_->spec().lr_scheduler_spec);
    policy_ = utils::create_action_policy(spec_.action_policy);
    explore_var_scheduler_ = utils::VarScheduler(spec_.explore_spec);
    entropy_coef_scheduler_ = utils::VarScheduler(spec_.entropy_spec);
}

torch::Tensor Algorithm::train()
{
    LAB_UNIMPLEMENTED;
    return torch::Tensor();
}

void Algorithm::update()
{
    LAB_UNIMPLEMENTED;
}

torch::IValue Algorithm::act(const torch::Tensor& state)
{
    LAB_UNIMPLEMENTED;
    return torch::IValue();
}

torch::Tensor Algorithm::sample()
{
    LAB_UNIMPLEMENTED;
    return torch::Tensor();
}

torch::Tensor Algorithm::calc_pdparam(torch::Tensor x)
{
    LAB_UNIMPLEMENTED;
    return torch::Tensor();
}

void Algorithm::save(torch::serialize::OutputArchive& archive) const
{

}

void Algorithm::load(torch::serialize::InputArchive& archive)
{

}

torch::serialize::OutputArchive& operator<<(torch::serialize::OutputArchive& archive, const std::shared_ptr<Algorithm>& algo)
{
    return archive;
}

torch::serialize::InputArchive& operator>>(torch::serialize::InputArchive& archive, const std::shared_ptr<Algorithm>& algo)
{
    return archive;
}

}

}