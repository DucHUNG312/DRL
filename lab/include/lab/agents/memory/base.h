#pragma once

#include "lab/core.h"
#include "lab/utils/spectypes.h"
#include "lab/envs/base.h"
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
    Memory(const std::shared_ptr<Body>& body, const utils::MemorySpec& spec)
        : body_(body), spec_(std::move(spec))
    {
        reset();
    }
    LAB_DEFAULT_CONSTRUCT(Memory);

    void reset()
    {
        LAB_UNIMPLEMENTED;    
    }

    void update(const envs::StepResult& result)
    {
        LAB_UNIMPLEMENTED;
    }

    ExperienceDict sample()
    {
        LAB_UNIMPLEMENTED;
        return ExperienceDict();
    }

    void save(torch::serialize::OutputArchive& archive) const
    {

    }

    void load(torch::serialize::InputArchive& archive)
    {

    }
private:
    void add_experience(const envs::StepResult& result)
    {
        LAB_UNIMPLEMENTED;
    }
};

LAB_FORCE_INLINE torch::serialize::OutputArchive& operator<<(torch::serialize::OutputArchive& archive, const std::shared_ptr<Memory>& memory)
{
    return archive;
}

LAB_FORCE_INLINE torch::serialize::InputArchive& operator>>(torch::serialize::InputArchive& archive, const std::shared_ptr<Memory>& memory)
{
    return archive;
}

}
}