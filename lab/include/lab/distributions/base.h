#pragma once

#include "lab/core.h"

namespace lab
{
namespace distributions
{

class Distribution 
{
    LAB_ARG(bool, has_rsample) = false;
    LAB_ARG(torch::IntArrayRef, batch_shape);
    LAB_ARG(torch::Tensor, mean);
    LAB_ARG(torch::Tensor, variance);
    LAB_ARG(torch::Tensor, stddev);
public:
    Distribution(torch::IntArrayRef batch_shape);
    LAB_DEFAULT_CONSTRUCT(Distribution);

    virtual std::vector<int64_t> extended_shape(torch::IntArrayRef sample_shape = {});

    virtual torch::Tensor sample(torch::IntArrayRef sample_shape = {}) = 0;

    virtual torch::Tensor rsample(torch::IntArrayRef sample_shape = {}) = 0;

    virtual torch::Tensor sample_n(int64_t n) = 0;

    virtual torch::Tensor log_prob(const torch::Tensor& value) = 0;

    virtual torch::Tensor cdf(const torch::Tensor& value) = 0;

    virtual torch::Tensor icdf(const torch::Tensor& value) = 0;

    virtual torch::Tensor entropy() = 0;

};

}

}