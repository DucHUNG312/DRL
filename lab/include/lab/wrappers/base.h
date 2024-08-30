#pragma once

#include "lab/core.h"
#include "lab/spaces/base.h"

namespace lab
{
namespace wrappers
{

template<typename Env>
class Wrapper
{
private:
    spaces::SpaceHolder<Env> env_;
public:
    Wrapper(const spaces::SpaceHolder<Env>& env)
        : env_(std::move(env))
    {}

    void reset(uint64_t seed = 0)
    {
        env_->reset(seed);
    }

    void step(const torch::IValue& action)
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

    spaces::SpaceHolder<Env>& unwrapped()
    {
        return env_;
    }
};

template<typename Env>
class ObservationWrapper : public Wrapper<Env>
{
public:
    using Wrapper<Env>::Wrapper;

    void reset(uint64_t seed = 0)
    {
        this->unwrapped()->reset(seed);
        observation_ = this->unwrapped()->result_.state;
    }

    void step(const torch::IValue& act)
    {
        this->unwrapped()->step(act);
        observation_ = this->unwrapped()->result_.state;
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
    using Wrapper<Env>::Wrapper;

    void step(const torch::IValue& act)
    {
        this->unwrapped()->step(act);
        reward_ = this->unwrapped()->result_.reward;
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
    using Wrapper<Env>::Wrapper;

    void step(const torch::IValue& act)
    {
        action_ = act;
        this->unwrapped()->step(action_);
    }

    torch::IValue action(torch::IValue& act)
    {
        LAB_UNIMPLEMENTED;
        return torch::IValue();
    }

    torch::IValue reverse_action(torch::IValue& act)
    {
        LAB_UNIMPLEMENTED;
        return action(act);
    }
protected:
    torch::IValue action_;
};

}
}