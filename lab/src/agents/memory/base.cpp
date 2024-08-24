#include "lab/agents/memory/base.h"

namespace lab
{

namespace agents
{

Memory::Memory(const utils::MemorySpec& spec)
    : spec_(spec) {}

void Memory::reset() {};
void Memory::update() {};
void Memory::sample() {};

}

}