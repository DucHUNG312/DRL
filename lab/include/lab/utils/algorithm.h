#pragma once

#include "lab/common/common.h"
#include "lab/utils/typetraits.h"

namespace lab {
namespace utils {

using Algorithms = types_t<agents::Reinforce>;
constexpr named_factory_t<std::shared_ptr<agents::Algorithm>, shared_ptr_maker, Algorithms> AlgorithmFactory;

std::shared_ptr<agents::Algorithm> create_algorithm(const AlgorithmSpec& spec);

} // namespace utils

} // namespace lab
