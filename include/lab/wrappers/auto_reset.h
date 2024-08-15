#pragma once

#include "lab/wrappers/base.h"

namespace lab
{
namespace wrappers
{

template<typename ObsSpace, typename ActSpace>
class AutoResetWrapper : public Wrapper<ObsSpace, ActSpace>
{
public:
    using EnvType = envs::Env<ObsSpace, ActSpace>;
    using ObsType = typename ObsSpace::Type;
    using ActType = typename ActSpace::Type;
    using StepResultType = utils::StepResult<ObsType>;
public:
    AutoResetWrapper(const EnvType& env)
        : Wrapper<ObsSpace, ActSpace>(env)
    {}

    virtual StepResultType step(const ActType& action) override
    {
        StepResultType result = this->env()->step(action);
        ObsType new_obs;
        if (result.terminated || result.truncated)
        {
            new_obs = this->env()->reset();
        }

        result.next_state = new_obs;
        return result;
    }
};


}
}