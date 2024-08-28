#pragma once

#include "lab/distributions/base.h"

namespace lab
{
namespace distributions
{

class Cauchy : public Distribution
{
    LAB_ARG(torch::Tensor, loc);
    LAB_ARG(torch::Tensor, scale);
public:
    Cauchy(const torch::Tensor& loc, const torch::Tensor& scale);

    torch::Tensor rsample(torch::IntArrayRef sample_shape) override;

    torch::Tensor log_prob(const torch::Tensor& value) override;

    torch::Tensor cdf(const torch::Tensor& value) override;

    torch::Tensor icdf(const torch::Tensor& value) override;

    torch::Tensor entropy() override;
};

}

}