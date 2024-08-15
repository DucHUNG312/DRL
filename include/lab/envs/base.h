#pragma once

#include "lab/core.h"
#include "lab/spaces/spaces.h"
#include "lab/utils/utils.h"

namespace lab
{
namespace envs
{
template<typename ObsSpace, typename ActSpace>
class Env : public c10::intrusive_ptr_target
{
public:
    using ObsType = typename ObsSpace::Type;
    using ActType = typename ActSpace::Type;

    LAB_ARG(ObsSpace, observation_space);
    LAB_ARG(ObsType, state);
    LAB_ARG(ActSpace, action_space);
    LAB_ARG(utils::EnvOptions, env_options);
    LAB_ARG(utils::Rand, rand);
public:
    Env(uint64_t seed = 0)
    {
        set_seed(seed);
    }

    Env(const utils::EnvOptions& env_options = {})
        : env_options_(env_options)
    {
        set_seed(env_options.seed);
    }

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

    virtual utils::StepResult<ObsType> step(const ActType& action) = 0;

    virtual void render() = 0;

    virtual void close() = 0;

    virtual c10::intrusive_ptr<Env> unwrapped() = 0;

    void set_seed(uint64_t seed = 0)
    {
        rand_.set_seed(seed);
    }

    void enable_rendering(const std::string& mode = "human")
    {
        if(mode != "None" || mode != "human") env_options_.render_mode = "human";
        else env_options_.render_mode = mode;
    }
private:
    void copy_from(const Env& other)
    {
        observation_space_ = other.observation_space_;
        state_ = other.state_;
        action_space_ = other.action_space_;
        env_options_ = other.env_options_;
        rand_ = other.rand_;
    }

    void move_from(Env&& other) noexcept
    {
        observation_space_ = std::move(other.observation_space_);
        state_ = std::move(other.state_);
        action_space_ = std::move(other.action_space_);
        env_options_ = std::move(other.env_options_);
        rand_ = std::move(other.rand_);
    }
};

using FiniteEnv = Env<spaces::Discrete, spaces::Discrete>;
using ContinuousEnv = Env<spaces::Box, spaces::Box>;
using ContinuousStateEnv = Env<spaces::Box, spaces::Discrete>;
using ContinuousActionEnv = Env<spaces::Discrete, spaces::Box>;

using FiniteResult = utils::StepResult<int64_t>;
using ContinuousResult = utils::StepResult<torch::Tensor>;

}
}
