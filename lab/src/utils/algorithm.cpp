#include "lab/utils/algorithm.h"
#include "lab/agents/algorithms/base.h"
#include "lab/agents/algorithms/reinforce.h"
#include "lab/utils/spec.h"

namespace lab {

namespace utils {

std::shared_ptr<agents::Algorithm> create_algorithm(const AlgorithmSpec& spec) {
  return AlgorithmFactory(spec.name, spec);
}

} // namespace utils

} // namespace lab