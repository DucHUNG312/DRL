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

    virtual void step(const ActType& action) override
    {
        this->env()->step(action);
        if (this->env()->result().terminated || this->env()->result().truncated)
            this->env()->reset();
    }
};


}
}