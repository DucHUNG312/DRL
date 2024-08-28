#pragma once

#include "lab/agents/memory/base.h"

namespace lab
{

namespace agents
{

class OnPolicyReplay : public Memory
{
public:
    using Memory::Memory;
    using Memory::ExperienceDict;

    void reset();

    void update(const envs::StepResult& result);

    ExperienceDict sample();
private:
    void add_experience(const envs::StepResult& result);
};

}

}