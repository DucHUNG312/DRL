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

    torch::Tensor act(const torch::Tensor& state);

    torch::Tensor update(const envs::StepResult& result);

    void close();

    void save(torch::serialize::OutputArchive& archive) const;

    void load(torch::serialize::InputArchive& archive);
};

torch::serialize::OutputArchive& operator<<(torch::serialize::OutputArchive& archive, const std::shared_ptr<Agent>& agent);

torch::serialize::InputArchive& operator>>(torch::serialize::InputArchive& archive, const std::shared_ptr<Agent>& agent);


}
}