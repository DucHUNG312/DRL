#pragma once

#include "lab/wrappers/base.h"

namespace lab
{
namespace wrappers
{

template<typename ObsSpace, typename ActSpace>
class TimeLimit : public Wrapper<ObsSpace, ActSpace>
{
public:
    using EnvType = envs::Env<ObsSpace, ActSpace>;
    using ObsType = typename ObsSpace::Type;
    using ActType = typename ActSpace::Type;
    using StepResultType = utils::StepResult<ObsType>;

    LAB_ARG(int64_t, max_episode_steps);
    LAB_ARG(int64_t, elapsed_steps);
public:

    TimeLimit(const EnvType& env, int64_t max_episode_steps = -1)
        : Wrapper<ObsSpace, ActSpace>(env)
    {
        if(max_episode_steps == -1 && this->env()->env_options().max_episode_steps != -1)
            max_episode_steps =  this->env()->env_options().max_episode_steps;
        else if(this->env()->env_options().max_episode_steps == -1)
            this->env()->env_options().max_episode_steps = max_episode_steps;
        max_episode_steps_ = max_episode_steps;
        elapsed_steps_ = 0;
    }

    virtual void reset(uint64_t seed = 0) override
    {
        elapsed_steps_ = 0;
        this->env()->reset(seed);
    }

    virtual void step(const ActType& action) override
    {
        this->env()->step(action);
        elapsed_steps_ += 1;

        if(elapsed_steps_ >= max_episode_steps_)
            this->env()->result().truncated = true;
    }
};


}
}