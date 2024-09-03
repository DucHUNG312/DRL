#pragma once

#include "lab/core.h"
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

    torch::Tensor act();

    void update();

    void close();

    void save(torch::serialize::OutputArchive& archive) const;

    void load(torch::serialize::InputArchive& archive);

    void reset_env();

    double get_total_reward() const;

    bool is_env_terminated() const;

    void step(const torch::Tensor& act);

    torch::Tensor get_result_state() const;

    std::shared_ptr<utils::Clock> get_env_clock() const;
};

torch::serialize::OutputArchive& operator<<(torch::serialize::OutputArchive& archive, const std::shared_ptr<Agent>& agent);

torch::serialize::InputArchive& operator>>(torch::serialize::InputArchive& archive, const std::shared_ptr<Agent>& agent);


}
}