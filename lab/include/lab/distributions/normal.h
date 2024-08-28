#pragma once

#include "lab/distributions/exp_family.h"

namespace lab
{
namespace distributions
{

class Normal : public ExponentialFamily
{
    LAB_ARG(torch::Tensor, loc);
    LAB_ARG(torch::Tensor, scale);
public:
    Normal(const torch::Tensor& loc, const torch::Tensor& scale);

    torch::Tensor sample(torch::IntArrayRef sample_shape) override;

    torch::Tensor rsample(torch::IntArrayRef sample_shape) override;

    torch::Tensor log_prob(const torch::Tensor& value) override;

    torch::Tensor cdf(const torch::Tensor& value) override;

    torch::Tensor icdf(const torch::Tensor& value) override;

    torch::Tensor entropy() override;

    torch::Tensor log_normalizer(torch::TensorList params) override;
};

}

}