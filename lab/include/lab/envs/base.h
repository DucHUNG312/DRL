#pragma once

#include "lab/core.h"
#include "lab/spaces/spaces.h"
#include "lab/utils/utils.h"
#include <renderer/renderer.h>

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
    LAB_ARG(utils::StepResult<ObsType>, result);
    LAB_ARG(ActSpace, action_space);
    LAB_ARG(utils::EnvSpec, env_spec);
    LAB_ARG(utils::Rand, rand);
    LAB_ARG(utils::Clock, clock);
public:
    Env(uint64_t seed = 0)
    {
        set_seed(seed);
    }

    Env(const utils::EnvSpec& env_spec = {})
        : env_spec_(env_spec)
    {
        set_seed(env_spec.seed);
    }

    virtual void reset(uint64_t seed = 0) = 0;

    virtual void step(const ActType& action) = 0;

    virtual void close() = 0;

    virtual void render() = 0;

    virtual c10::intrusive_ptr<Env> unwrapped() = 0;

    void set_seed(uint64_t seed = 0)
    {
        rand_.set_seed(seed);
    }

    void enable_rendering()
    {
        env_spec_.renderer.enabled = true;
        render::Renderer::init();
        render();
    }

    int64_t get_observable_dim()
    {
        int64_t state_dim;
        if (std::is_same_v<ObsSpace, spaces::Box>)
            state_dim = result().state[0]; 
        else if (std::is_same_v<ObsSpace, spaces::Discrete>)
            state_dim = 1; 
        
        return state_dim;
    }

    int64_t get_action_dim()
    {
        int64_t action_dim;
        if (std::is_same_v<ActSpace, spaces::Box>)
        {
            LAB_CHECK_EQ(action_space_.shape().size(), 1);
            action_dim = action_space_.shape()[0];
        }
        else if (std::is_same_v<ActSpace, spaces::Discrete>)
            action_dim = action_space_.n();
        else
            LAB_LOG_FATAL("action_space not recognized");
        return action_dim;
    }

    bool is_discrete() const
    {
        return std::is_same_v<ActSpace, spaces::Discrete>;
    }

    void load_spec(const utils::EnvSpec& spec)
    {

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