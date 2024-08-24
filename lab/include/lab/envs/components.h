#pragma once

#include "lab/core.h"

namespace lab
{

namespace envs
{
    
struct CartPoleActionComponent
{
    int64_t action; // 0 is left; 1 is right

    LAB_DEFAULT_CONSTRUCT(CartPoleActionComponent);
    CartPoleActionComponent(int64_t action)
        : action(action) {}
};

struct StepResultComponent
{
    std::vector<double> state;
    double reward = 0;
    bool terminated = false; 
    bool truncated = false;

    LAB_DEFAULT_CONSTRUCT(StepResultComponent);
    StepResultComponent(const std::vector<double>& state, double reward, bool terminated, bool truncated)
        : state(state), reward(reward), terminated(terminated), truncated(truncated) {}
};

}

}