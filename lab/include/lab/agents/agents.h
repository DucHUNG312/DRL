#pragma once

#include "lab/agents/algorithms/algorithms.h"
#include "lab/agents/memory/memory.h"
#include "lab/agents/net/net.h"
#include "lab/envs/envs.h"

namespace lab
{
namespace agents
{

class Agent;

class Body
{
    LAB_ARG(std::shared_ptr<envs::Env>, env);
    LAB_ARG(std::shared_ptr<Agent>, agent);
    LAB_ARG(Memory, memory);
    LAB_ARG(utils::DataFrame, train_df);
public:
    Body(const std::shared_ptr<envs::Env>& env);
    LAB_DEFAULT_CONSTRUCT(Body); 
};

class Agent : public std::enable_shared_from_this<Agent>
{
    //LAB_ARG(utils::AgentSpec, spec);
    LAB_ARG(Algorithm, algorithm);
    LAB_ARG(std::shared_ptr<Body>, body);
public:
    Agent(const std::shared_ptr<Body>& body);
    LAB_DEFAULT_CONSTRUCT(Agent);

    template<typename ActType>
    ActType act(const ActType& action);
};

template<typename ActType>
ActType Agent::act(const ActType& action)
{
    torch::NoGradGuard no_grad;
    return algorithm_.act(action);
}

}
}