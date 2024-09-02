#pragma once

#include "lab/core.h"
#include "lab/utils/spec.h"
#include "lab/utils/typetraits.h"

namespace lab
{

namespace agents
{
class Algorithm;
class Reinforce;
}

namespace utils
{

using Algorithms = types_t<agents::Reinforce>;
constexpr named_factory_t<std::shared_ptr<agents::Algorithm>, shared_ptr_maker, Algorithms> AlgorithmFactory;

std::shared_ptr<agents::Algorithm> create_algorithm(const AlgorithmSpec& spec);

}

}

