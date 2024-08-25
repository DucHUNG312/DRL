#pragma once

#include "lab/core.h"
#include "lab/spaces/base.h"
#include "lab/utils/rand.h"
#include "lab/utils/spec.h"

#include <renderer/renderer.h>

namespace lab
{
namespace envs
{

struct StepResult
{
    torch::Tensor state;
    torch::Tensor next_state;
    torch::IValue action;
    double reward = 0;
    bool terminated = false; 
    bool truncated = false;

    StepResult(
        const torch::Tensor& state, 
        const torch::IValue& action,
        double reward, 
        bool terminated, 
        bool truncated);
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

    void step(torch::IValue act);

    torch::IValue sample();

    void close();

    void render();

    void enable_rendering();

    void save(torch::serialize::OutputArchive& archive) const;

    void load(torch::serialize::InputArchive& archive);

    const StepResult& get_result() const;

    const utils::EnvSpec& get_env_spec() const;

    std::shared_ptr<spaces::Space>& get_observation_spaces();

    std::shared_ptr<spaces::Space>& get_action_spaces();

    int64_t get_state_dim() const;

    int64_t get_action_dim() const;

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
