#pragma once

#include "lab/distributions/dirichlet.h"
#include "lab/distributions/exp_family.h"

namespace lab {
namespace distributions {

class Beta : public ExponentialFamily {
  LAB_ARG(torch::Tensor, concentration0);
  LAB_ARG(torch::Tensor, concentration1);
  LAB_ARG(Dirichlet, dirichlet);

 public:
  Beta(const torch::Tensor& concentration0, const torch::Tensor& concentration1);

  torch::Tensor rsample(torch::IntArrayRef sample_shape = {}) override;

  torch::Tensor log_prob(const torch::Tensor& value) override;

  torch::Tensor log_normalizer(torch::TensorList params) override;
};

} // namespace distributions

} // namespace lab