#pragma once

#include "lab/distributions/exp_family.h"

namespace lab
{
namespace distributions
{

class Bernoulli : public ExponentialFamily
{
    LAB_ARG(bool , is_logits) = false;
    LAB_ARG(torch::Tensor, params);
public:
    Bernoulli(const torch::Tensor& in, bool is_logits = false);

    torch::Tensor logits();

    torch::Tensor probs();

    torch::IntArrayRef params_shape();

    torch::Tensor sample(torch::IntArrayRef sample_shape = {}) override;

    torch::Tensor log_prob(const torch::Tensor& value) override;

    torch::Tensor entropy() override;

    torch::Tensor log_normalizer(torch::TensorList params) override;
};

}

}