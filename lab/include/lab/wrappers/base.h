#pragma once

#include "lab/core.h"
#include "lab/envs/envs.h"

namespace lab
{
namespace wrappers
{

template<typename Env>
class Wrapper
{
    using ActType = typename Env::ActType;
protected:
    c10::intrusive_ptr<Env> env_;
public:
    Wrapper(const c10::intrusive_ptr<Env>& env)
        : env_(std::move(env))
    {}

    void reset(uint64_t seed = 0)
    {
        env_->reset(seed);
    }

    void step(const ActType& action)
    {
        env_->step(action);
    }

    void render()
    {
        env_->render();
    }

    void close()
    {
        env_->close();
    }

    c10::intrusive_ptr<Env> unwrapped()
    {
        return env_;
    }
};

template<typename Env>
class ObservationWrapper : public Wrapper<Env>
{
public:
    using ActType = typename Env::ActType;
    using Wrapper<Env>::Wrapper;

    void reset(uint64_t seed = 0)
    {
        this->env_->reset(seed);
        observation_ = this->env_->result_.state;
    }

    void step(const ActType& act)
    {
        this->env_->step(act);
        observation_ = this->env_->result_.state;
    }

    torch::Tensor observation(torch::Tensor& obs)
    {
        LAB_UNIMPLEMENTED;
        return torch::Tensor();
    }
protected:
    torch::Tensor observation_;
};

template<typename Env>
class RewardWrapper : public Wrapper<Env>
{
public:
    using ActType = typename Env::ActType;
    using Wrapper<Env>::Wrapper;

    void step(const ActType& act)
    {
        this->env_->step(act);
        reward_ = this->env_->result_.reward;
    }

    double reward(double r)
    {
        LAB_UNIMPLEMENTED;
        return 0;
    }
protected:
    double reward_;
};

template<typename Env>
class ActionWrapper : public Wrapper<Env>
{
public:
    using ActType = typename Env::ActType;
    using Wrapper<Env>::Wrapper;

    void step(const ActType& act)
    {
        action_ = act;
        this->env_->step(action_);
    }

    ActType action(ActType& act)
    {
        LAB_UNIMPLEMENTED;
        return ActType();
    }

    ActType reverse_action(ActType& act)
    {
        LAB_UNIMPLEMENTED;
        return action(act);
    }
protected:
    ActType action_;
};

}
}