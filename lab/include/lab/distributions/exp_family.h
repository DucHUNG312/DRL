#pragma once

#include "lab/distributions/base.h"

namespace lab {
namespace distributions {

class ExponentialFamily : public Distribution {
  LAB_ARG(torch::TensorList, natural_params);
  LAB_ARG(torch::Tensor, mean_carrier_measure);

 public:
  using Distribution::Distribution;

  virtual torch::Tensor log_normalizer(torch::TensorList params) = 0;

  virtual torch::Tensor sample(torch::IntArrayRef sample_shape = {}) override;

  virtual torch::Tensor rsample(torch::IntArrayRef sample_shape = {}) override;

  virtual torch::Tensor sample_n(int64_t n) override;

  virtual torch::Tensor log_prob(const torch::Tensor& value) override;

  virtual torch::Tensor cdf(const torch::Tensor& value) override;

  virtual torch::Tensor icdf(const torch::Tensor& value) override;

  virtual torch::Tensor entropy() override;
};

} // namespace distributions

} // namespace lab
