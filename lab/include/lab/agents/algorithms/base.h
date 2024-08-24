#pragma once

#include "lab/agents/base.h"

namespace lab
{
namespace agents
{

class Algorithm
{
public:
    LAB_DEFAULT_CONSTRUCT(Algorithm);

    template<typename ActType>
    ActType act(const ActType& action);
};

template<typename ActType>
ActType Algorithm::act(const ActType& action)
{
    return action;
}

}
}