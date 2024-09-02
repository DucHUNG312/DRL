#pragma once

#include "lab/agents/memory/base.h"

namespace lab
{

namespace agents
{

class OnPolicyReplay : public Memory
{
public:
    static constexpr const char* name = "OnPolicyReplay";
public:
    using Memory::Memory;
    using Memory::ExperienceDict;

    void update(const utils::StepResult& result) override;

    ExperienceDict sample() override;
private:
    void add_experience(const utils::StepResult& result) override;
};

}

}