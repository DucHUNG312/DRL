#pragma once

#include "lab/common/common.h"
#include "lab/envs/base.h"
#include "lab/utils/spec.h"
#include "lab/agents/memory/base.h"
#include "lab/agents/algorithms/base.h"

namespace lab
{
namespace agents
{

class Body
{
    LAB_ARG(std::shared_ptr<envs::Env>, env);
    LAB_ARG(std::shared_ptr<Memory>, memory);
    LAB_ARG(std::shared_ptr<Algorithm>, algorithm);
    LAB_ARG(utils::BodySpec, spec);
public:
    using ExperienceDict = torch::Dict<std::string, torch::List<torch::IValue>>;

    Body(const std::shared_ptr<envs::Env>& env, const utils::BodySpec& spec);
    LAB_DEFAULT_CONSTRUCT(Body);

    void update();

    torch::Tensor act();

    torch::Tensor train();

    ExperienceDict sample();

    void save(torch::serialize::OutputArchive& archive) const;

    void load(torch::serialize::InputArchive& archive);

    torch::Tensor get_loss() const;

    void close_env();

    void reset_env();

    double get_total_reward() const;

    bool is_env_terminated() const;

    void step(const torch::Tensor& act);

    torch::Tensor get_result_state() const;

    std::shared_ptr<utils::Clock> get_env_clock() const;
private:
    void init_algorithm_specs(const utils::AlgorithmSpec& spec);

    void init_memory_spec(const utils::MemorySpec& spec);
};

torch::serialize::OutputArchive& operator<<(torch::serialize::OutputArchive& archive, const std::shared_ptr<Body>& body);

torch::serialize::InputArchive& operator>>(torch::serialize::InputArchive& archive, const std::shared_ptr<Body>& body);

}

}