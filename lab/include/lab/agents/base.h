#pragma once

#include "lab/agents/body.h"
#include "lab/utils/spec.h"

namespace lab
{
namespace agents
{

class Agent 
{
    LAB_ARG(utils::AgentSpec, spec);
    LAB_ARG(std::shared_ptr<Body>, body);
public:
    Agent(const std::shared_ptr<Body>& body, const utils::AgentSpec& spec);
    LAB_DEFAULT_CONSTRUCT(Agent);

    torch::IValue act(const torch::Tensor& state)
    {
        torch::NoGradGuard no_grad;
        LAB_CHECK(body_ != nullptr);
        return body_->act(std::move(state));
    }

    torch::Tensor update(const envs::StepResult& result)
    {
        body_->update(std::move(result));
        torch::Tensor loss = body_->train();
        if(loss.defined())
            body_->loss(loss);
        return loss;
    }

    void save(torch::serialize::OutputArchive& archive) const
    {

    }

    void load(torch::serialize::InputArchive& archive)
    {

    }
};

LAB_FORCE_INLINE torch::serialize::OutputArchive& operator<<(torch::serialize::OutputArchive& archive, const std::shared_ptr<Agent>& agent)
{
    return archive;
}

LAB_FORCE_INLINE torch::serialize::InputArchive& operator>>(torch::serialize::InputArchive& archive, const std::shared_ptr<Agent>& agent)
{
    return archive;
}


}
}