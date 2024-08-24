#pragma once

#include "lab/core.h"
#include "lab/spaces/spaces.h"
#include "lab/utils/utils.h"
#include <renderer/renderer.h>

namespace lab
{
namespace envs
{

// class Env : public renderer::Scene
// {
// public:
//     using ObsType = typename ObsSpace::Type;
//     using ActType = typename ActSpace::Type;

//     LAB_ARG(ObsSpace, observation_space);
//     LAB_ARG(utils::StepResult<ObsType>, result);
//     LAB_ARG(ActSpace, action_space);
//     LAB_ARG(utils::EnvSpec, env_spec);
//     LAB_ARG(utils::Rand, rand);
//     LAB_ARG(utils::Clock, clock);

//     torch::OrderedDict<std::string, std::shared_ptr<spaces::Space>> obs_spaces_;
//     torch::OrderedDict<std::string, std::shared_ptr<spaces::Space>> act_spaces_;
//     ObservableState state;
// public:
//     Env() = default;
//     explicit Env(std::string name);
//     Env(const Env& other) = default;
//     Env& operator=(const Env& other) = default;
//     Env(Env&& other) noexcept = default;
//     Env& operator=(Env&& other) noexcept = default;
//     virtual ~Env() = default;

//     virtual std::shared_ptr<Env> clone(const std::optional<torch::Device>& device = std::nullopt) const;

//     void save(torch::serialize::OutputArchive& archive) const;

//     void load(torch::serialize::InputArchive& archive);

//     template <typename EnvType>
//     std::shared_ptr<EnvType> register_action_space(std::string name, std::shared_ptr<EnvType> space);

//     template <typename EnvType>
//     std::shared_ptr<EnvType> register_observation_space(std::string name, utils::SpaceHolder<EnvType> space_holder);

//     torch::Tensor& register_parameter(std::string name, torch::Tensor tensor, bool requires_grad = false);
    
//     torch::Tensor& get_parameter(std::string name);

//     bool is_serializable() const;

//     virtual void pretty_print(std::ostream& stream) const;

//     std::ostream& operator<<(std::ostream& stream);


//     Env(const utils::EnvSpec& env_spec = {})
//         : env_spec_(env_spec)
//     {
//         //reset(env_spec.seed);
//     }

//     virtual void reset(uint64_t seed = 0) = 0;

//     template<typename ActSpace>
//     virtual void step(const ActType& action) = 0;

//     virtual void close() = 0;

//     virtual void render() = 0;

//     virtual c10::intrusive_ptr<Env> unwrapped() = 0;

//     void enable_rendering()
//     {
//         env_spec_.renderer.enabled = true;
//         render::Renderer::init();
//         render();
//     }

//     int64_t get_observable_dim()
//     {
//         int64_t state_dim;
//         if (std::is_same_v<ObsSpace, spaces::Box>)
//             state_dim = result().state[0]; 
//         else if (std::is_same_v<ObsSpace, spaces::Discrete>)
//             state_dim = 1; 
        
//         return state_dim;
//     }

//     int64_t get_action_dim()
//     {
//         int64_t action_dim;
//         if (std::is_same_v<ActSpace, spaces::Box>)
//         {
//             LAB_CHECK_EQ(action_space_.shape().size(), 1);
//             action_dim = action_space_.shape()[0];
//         }
//         else if (std::is_same_v<ActSpace, spaces::Discrete>)
//             action_dim = action_space_.n();
//         else
//             LAB_LOG_FATAL("action_space not recognized");
//         return action_dim;
//     }

//     bool is_discrete() const
//     {
//         return std::is_same_v<ActSpace, spaces::Discrete>;
//     }

//     void load_spec(const utils::EnvSpec& spec)
//     {

//     }
// };

// using FiniteEnv = Env<spaces::Discrete, spaces::Discrete>;
// using ContinuousEnv = Env<spaces::Box, spaces::Box>;
// using ContinuousStateEnv = Env<spaces::Box, spaces::Discrete>;
// using ContinuousActionEnv = Env<spaces::Discrete, spaces::Box>;

// using FiniteResult = utils::StepResult<int64_t>;
// using ContinuousResult = utils::StepResult<torch::Tensor>;

/*

    register attribute for environment: gravity, tau, action space, observable space, theta_threshold_radians, reset_low, reset_high
    register object for environment: cart, pole                 
    register attribute for object: gravity, mass, length
    register component for manipulate object state: force
    init object and environment state
    define a reward mechanism, bind it to env and object state
    monorting env and object state
    

    modify a state in one object in env can take an effect on another object state in env
    => we need a state manager

    // we should use ECS 

*/

struct StepResult
{
    torch::Tensor state;
    double reward = 0;
    bool terminated = false; 
    bool truncated = false;

    StepResult(const torch::Tensor& state, double reward, bool terminated, bool truncated);
    LAB_DEFAULT_CONSTRUCT(StepResult);

    void pretty_print(std::ostream& stream, const std::string& indentation) const;
};

LAB_FORCE_INLINE std::ostream& operator<<(std::ostream& stream, const StepResult& result)
{
    result.pretty_print(stream, "");
    return stream;
}
class Env : public renderer::Scene
{
protected:
    utils::EnvSpec env_spec_;
    utils::Rand rand_;
    // utils::Clock clock_;
    StepResult result_;
    std::shared_ptr<spaces::Space> observation_spaces_;
    std::shared_ptr<spaces::Space> action_spaces_;
public:
    explicit Env(const utils::EnvSpec& env_spec);
    LAB_DEFAULT_CONSTRUCT(Env);

    void reset(uint64_t seed = 0);

    void step(int64_t act);

    int64_t sample();

    void close();

    void render();

    void enable_rendering();

    void save(torch::serialize::OutputArchive& archive) const;

    void load(torch::serialize::InputArchive& archive);

    const StepResult& get_result() const;

    const utils::EnvSpec& get_env_spec() const;

    std::shared_ptr<spaces::Space>& get_observation_spaces();

    std::shared_ptr<spaces::Space>& get_action_spaces();

    bool is_serializable() const;

    void pretty_print(std::ostream& stream, const std::string& indentation) const;
};

LAB_FORCE_INLINE std::ostream& operator<<(std::ostream& stream, const Env& env)
{
    env.pretty_print(stream, "");
    return stream;
}

torch::serialize::OutputArchive& operator<<(torch::serialize::OutputArchive& archive, const std::shared_ptr<Env>& env);

torch::serialize::InputArchive& operator>>(torch::serialize::InputArchive& archive, const std::shared_ptr<Env>& env);

}
}
