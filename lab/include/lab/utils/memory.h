#pragma once

#include "lab/common/common.h"
#include "lab/utils/typetraits.h"

namespace lab {

namespace utils {

using Memories = types_t<agents::OnPolicyReplay>;
constexpr named_factory_t<std::shared_ptr<agents::Memory>, shared_ptr_maker, Memories> MemoryFactory;

std::shared_ptr<agents::Memory> create_memory(const MemorySpec& spec);

} // namespace utils

} // namespace lab
