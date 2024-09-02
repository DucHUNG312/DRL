#include "lab/utils/memory.h"
#include "lab/agents/memory/base.h"
#include "lab/agents/memory/onpolicy.h"

namespace lab
{

namespace utils
{

std::shared_ptr<agents::Memory> create_memory(const MemorySpec& spec)
{
    return MemoryFactory(spec.name, spec);
}

}

}