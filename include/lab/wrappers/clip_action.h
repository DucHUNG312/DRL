#pragma once

#include "lab/wrappers/base.h"
#include "lab/spaces/box.h"

namespace lab
{
namespace wrappers
{

template<typename ObsSpace, typename ActSpace>
class ClipAction : public ActionWrapper<ObsSpace, ActSpace>
{
public:
    using EnvType = envs::Env<ObsSpace, ActSpace>;
    using ObsType = typename ObsSpace::Type;
    using ActType = typename ActSpace::Type;
    using StepResultType = utils::StepResult<ObsType>;
public:

    ClipAction(const EnvType& env)
        : Wrapper<ObsSpace, ActSpace>(env)
    {
        static_assert(std::is_same_v<ActSpace, spaces::Box>);

    }

    virtual ActType action(ActType& act) override
    {
        return torch::clip(act, this->env()->action_space().low(), this->env()->action_space().high());
    }
};


}
}