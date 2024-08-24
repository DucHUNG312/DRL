#pragma once

#include "lab/agents/base.h"
#include "lab/utils/spectypes.h"

namespace lab
{
namespace agents
{

class Memory
{
    LAB_ARG(utils::MemorySpec, spec);
public:
    using DataKeyDict = torch::Dict<std::string, torch::IValue>;

    Memory(const utils::MemorySpec& spec);
    LAB_DEFAULT_CONSTRUCT(Memory);

    void reset();
    void update();
    void sample();

};

}
}