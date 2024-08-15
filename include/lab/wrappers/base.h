#pragma once

#include "lab/core.h"
#include "lab/envs/envs.h"

namespace lab
{
namespace wrappers
{

template<typename ObsSpace, typename ActSpace>
class Wrapper : public envs::Env<ObsSpace, ActSpace>
{
public:
    using EnvType = envs::Env<ObsSpace, ActSpace>;
    using ObsType = typename ObsSpace::Type;
    using ActType = typename ActSpace::Type;
    using StepResultType = utils::StepResult<ObsType>;

    LAB_ARG(c10::intrusive_ptr<EnvType>, env);
public:
    Wrapper(const EnvType& env)
    {
        env_ = c10::make_intrusive<EnvType>(env);
    }

    virtual ObsType reset(uint64_t seed = 0) override
    {
        return env_->reset(seed);
    }

    virtual StepResultType step(const ActType& action) override
    {
        return env_->step(action);
    }

    virtual void render() override
    {
        env_->render();
    }

    virtual void close() override
    {
        env_->close();
    }

    virtual c10::intrusive_ptr<EnvType> unwrapped() override
    {
        return env_->unwrapped();
    }
};

template<typename ObsSpace, typename ActSpace>
class ObservationWrapper : public Wrapper<ObsSpace, ActSpace>
{
public:
    using ObsType = typename ObsSpace::Type;
    using ActType = typename ActSpace::Type;
    using StepResultType = utils::StepResult<ObsType>;

    LAB_ARG(ObsType, observation);
public:
    virtual ObsType reset(uint64_t seed = 0) override
    {
        observation_ = this->env()->reset(seed);
        return observation_;
    }

    virtual StepResultType step(const ActType& act) override
    {
        StepResultType result = this->env()->step(act);
        observation_ = result.next_state;
        return result;
    }

    virtual ObsType observation(ObsType& obs) = 0;
};

template<typename ObsSpace, typename ActSpace>
class RewardWrapper : public Wrapper<ObsSpace, ActSpace>
{
public:
    using ObsType = typename ObsSpace::Type;
    using ActType = typename ActSpace::Type;
    using StepResultType = utils::StepResult<ObsType>;

    LAB_ARG(double, reward);
public:
    virtual StepResultType step(const ActType& act) override
    {
        StepResultType result = this->env()->step(act);
        reward_ = result.reward;
        return result;
    }

    virtual double reward(double r) = 0;
};

template<typename ObsSpace, typename ActSpace>
class ActionWrapper : public Wrapper<ObsSpace, ActSpace>
{
public:
    using ObsType = typename ObsSpace::Type;
    using ActType = typename ActSpace::Type;
    using StepResultType = utils::StepResult<ObsType>;

    LAB_ARG(ActType, action);
public:
    virtual StepResultType step(const ActType& act) override
    {
        action_ = act;
        return this->env()->step(action_);
    }

    virtual ActType action(ActType& act) = 0;

    virtual ActType reverse_action(ActType& act)
    {
        LAB_UNIMPLEMENTED;
        return action(act);
    }
};

}
}