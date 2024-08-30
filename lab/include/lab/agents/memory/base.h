#pragma once

#include "lab/core.h"
#include "lab/envs/base.h"
#include "lab/utils/spec.h"
#include "lab/utils/policy.h"

namespace lab
{
namespace agents
{

class Body;

class Memory
{
public:
    using ExperienceDict = torch::Dict<std::string, torch::List<torch::IValue>>;

    LAB_ARG(std::shared_ptr<Body>, body);
    LAB_ARG(utils::MemorySpec, spec);
    LAB_ARG(envs::StepResult, most_recent);
    LAB_ARG(ExperienceDict, experiences);
    LAB_ARG(int64_t, size) = 0;
    LAB_ARG(int64_t, seen_size) = 0;
    LAB_ARG(bool, ready) = false;
    LAB_ARG(std::vector<std::string>, keys) = {"state", "new_state", "action", "reward", "terminated", "truncated"};
public:
    Memory(const std::shared_ptr<Body>& body, const utils::MemorySpec& spec);
    LAB_DEFAULT_CONSTRUCT(Memory);

    void reset();

    void update(const envs::StepResult& result);

    ExperienceDict sample();

    void save(torch::serialize::OutputArchive& archive) const;

    void load(torch::serialize::InputArchive& archive);
private:
    void add_experience(const envs::StepResult& result);
};

torch::serialize::OutputArchive& operator<<(torch::serialize::OutputArchive& archive, const std::shared_ptr<Memory>& memory);

torch::serialize::InputArchive& operator>>(torch::serialize::InputArchive& archive, const std::shared_ptr<Memory>& memory);

}
}