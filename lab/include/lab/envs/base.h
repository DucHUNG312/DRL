#pragma once

#include "lab/common/common.h"
#include "lab/spaces/any.h"
#include "lab/utils/rand.h"
#include "lab/utils/spec.h"
#include "lab/utils/clock.h"
#include "lab/utils/env.h"

//#include <renderer/renderer.h>

namespace lab
{
namespace envs
{

class Env /*: public renderer::Scene*/
{
protected:
    LAB_ARG(utils::EnvSpec, env_spec);
    LAB_ARG(std::shared_ptr<spaces::AnySpace>, observation_spaces);
    LAB_ARG(std::shared_ptr<spaces::AnySpace>, action_spaces);
    LAB_ARG(utils::Rand, rand);
    LAB_ARG(utils::StepResult, result);
    LAB_ARG(std::shared_ptr<utils::Clock>, clock);
    LAB_ARG(double, total_reward);
    LAB_ARG(bool, is_open) = false;
public:
    explicit Env(const utils::EnvSpec& env_spec);
    LAB_DEFAULT_CONSTRUCT(Env);

    virtual void reset(uint64_t seed = 0) = 0;

    virtual void step(const torch::Tensor& act) = 0;

    virtual torch::Tensor sample() = 0;

    virtual void close() = 0;

    virtual void render() = 0;

    void enable_rendering();

    bool done() const;

    bool is_venv() const;

    void save(torch::serialize::OutputArchive& archive) const;

    void load(torch::serialize::InputArchive& archive);

    const utils::StepResult& get_result() const;

    const utils::EnvSpec& get_env_spec() const;

    std::shared_ptr<spaces::AnySpace>& get_observation_spaces();

    std::shared_ptr<spaces::AnySpace>& get_action_spaces();

    int64_t get_state_dim() const;

    int64_t get_action_dim() const;

    bool is_serializable() const;

    void pretty_print(std::ostream& stream, const std::string& indentation) const;
};

std::ostream& operator<<(std::ostream& stream, const Env& env);

torch::serialize::OutputArchive& operator<<(torch::serialize::OutputArchive& archive, const std::shared_ptr<Env>& env);

torch::serialize::InputArchive& operator>>(torch::serialize::InputArchive& archive, const std::shared_ptr<Env>& env);

}
}
