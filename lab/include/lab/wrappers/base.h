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

    virtual void reset(uint64_t seed = 0) override
    {
        env_->reset(seed);
    }

    virtual void step(const ActType& action) override
    {
        env_->step(action);
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
    virtual void reset(uint64_t seed = 0) override
    {
        this->env()->reset(seed);
        observation_ = this->env()->result().state;
    }

    virtual void step(const ActType& act) override
    {
        this->env()->step(act);
        observation_ = this->env()->result().state;
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
    virtual void step(const ActType& act) override
    {
        this->env()->step(act);
        reward_ = this->env()->result().reward;
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
    virtual void step(const ActType& act) override
    {
        action_ = act;
        this->env()->step(action_);
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