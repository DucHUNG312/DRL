#pragma once

#include "lab/agents/base.h"

namespace lab
{
namespace agents
{

template<typename ObsSpace, typename ActSpace>
class Agent;

template<typename ObsSpace, typename ActSpace>
class Algorithm
{
public:
    using ObsType = typename ObsSpace::Type;
    using ActType = typename ActSpace::Type;
    using AgentType = Agent<ObsSpace, ActSpace>;

    LAB_ARG(AgentType, agent);
public:
    Algorithm(const AgentType& agent)
    {

    }

    ActType act(const ActType& action)
    {
        return action;
    }

};

}
}