#pragma once

#include "lab/spaces/any.h"
#include "lab/spaces/base.h"

namespace lab {
namespace spaces {

struct DiscreteOptions {
  DiscreteOptions(int64_t n, int64_t start);
  LAB_ARG(int64_t, n);
  LAB_ARG(int64_t, start);
};

class DiscreteImpl : public ClonableSpace<DiscreteImpl> {
 public:
  DiscreteImpl() = default;
  explicit DiscreteImpl(const DiscreteOptions& options_);
  explicit DiscreteImpl(int64_t n, int64_t start = 0) : DiscreteImpl(DiscreteOptions(n, start)) {}

  void reset() override;

  void pretty_print(std::ostream& stream) const override;

  torch::Tensor sample(/*const torch::Tensor& mask*/);

  bool contains(const torch::Tensor& x) const;

  // Overload for zero arguments
  bool contains() const;

 public:
  DiscreteOptions options;
  torch::Tensor n;
  torch::Tensor start;
};

LAB_SPACE(Discrete);

std::shared_ptr<DiscreteImpl> make_discrete_space_imp(int64_t n, int64_t start = 0);

Discrete make_discrete_space(int64_t n, int64_t start = 0);

std::shared_ptr<AnySpace> make_discrete_space_any(int64_t n, int64_t start = 0);

} // namespace spaces
} // namespace lab
