#include "lab/utils/memory.h"
#include "lab/agents/memory/base.h"
#include "lab/agents/memory/onpolicy.h"
#include "lab/utils/spec.h"

namespace lab::utils {

std::shared_ptr<agents::Memory> create_memory(const MemorySpec& spec) {
  return MemoryFactory(spec.name, spec);
}

} // namespace lab::utils
