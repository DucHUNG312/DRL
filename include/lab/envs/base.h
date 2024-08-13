#pragma once

#include "lab/core.h"
#include "lab/spaces/spaces.h"
#include "lab/utils/utils.h"

namespace lab
{
namespace envs
{

template<typename T>
class StepResult
{
    LAB_ARG(T, next_state);
    LAB_ARG(double, reward) = 0;
    LAB_ARG(bool, done) = false; 
public:
    StepResult() = default;
    StepResult(T next_state, double reward, bool done)
        : next_state_(next_state), reward_(reward), done_(done) {}
    ~StepResult() = default;
};

template<typename ObsSpace, typename ActSpace>
class Env
{
public:
    using ObsType = typename ObsSpace::Type;
    using ActType = typename ActSpace::Type;

    LAB_ARG(ObsSpace, observation_space);
    LAB_ARG(ObsSpace, state);
    LAB_ARG(ActSpace, action_space);
    LAB_ARG(utils::EnvSpec, spec);
    LAB_ARG(utils::Rand, rand);
public:
    Env() = default;

    Env(const utils::EnvSpec& spec)
        : spec_(spec)
    {}

    Env(const Env& other)
    {
        copy_from(other);
    }
    Env(Env&& other) noexcept
    {
        move_from(std::move(other));
    }
    virtual ~Env() = default;

    Env& operator=(const Env& other) 
    {
        if (this != &other) 
        {
            copy_from(other);
        }
        return *this;
    }

    Env& operator=(Env&& other) noexcept 
    {
        if (this != &other) 
        {
            move_from(std::move(other));
        }
        return *this;
    }

    virtual ObsType reset(uint64_t seed = 0) = 0;

    virtual StepResult<ObsType> step(ActType& action) = 0;

    virtual void render() = 0;

    virtual void close() = 0;

    virtual Env& unwrapped() = 0;

    void set_seed(uint64_t seed)
    {
        rand_.set_seed(seed);
    }
private:
    void copy_from(const Env& other)
    {
        observation_space_ = other.observation_space_;
        state_ = other.state_;
        action_space_ = other.action_space_;
        spec_ = other.spec_;
        rand_ = other.rand_;
    }

    void move_from(Env&& other) noexcept
    {
        observation_space_ = std::move(other.observation_space_);
        state_ = std::move(other.state_);
        action_space_ = std::move(other.action_space_);
        spec_ = std::move(other.spec_);
        rand_ = std::move(other.rand_);
    }
};

using FiniteEnv = Env<spaces::Discrete, spaces::Discrete>;
using ContinuousEnv = Env<spaces::Box, spaces::Box>;
using ContinuousStateEnv = Env<spaces::Box, spaces::Discrete>;
using ContinuousActionEnv = Env<spaces::Discrete, spaces::Box>;

}
}
