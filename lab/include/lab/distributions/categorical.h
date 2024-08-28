#pragma once

#include "lab/distributions/base.h"

namespace lab
{
namespace distributions
{

class Categorical : public Distribution 
{
    LAB_ARG(bool , is_logits) = false;
    LAB_ARG(torch::Tensor, params);
    LAB_ARG(int64_t, num_events);
public:
    Categorical(const torch::Tensor& in, bool is_logits = false);

    torch::Tensor logits();
    torch::Tensor probs();
    torch::IntArrayRef params_shape();

    torch::Tensor sample(torch::IntArrayRef sample_shape) override;
    torch::Tensor log_prob(const torch::Tensor& value) override;
    torch::Tensor entropy() override;
};

}

}