#pragma once

#include "lab/core.h"
namespace lab
{
namespace utils
{

class Distribution 
{
    LAB_ARG(bool, has_rsample) = false;
    LAB_ARG(bool, has_enumerate_support) = false;
    LAB_ARG(torch::IntArrayRef, batch_shape);
    LAB_ARG(torch::Tensor, mean);
    LAB_ARG(torch::Tensor, variance);
    LAB_ARG(torch::Tensor, stddev);
public:
    Distribution(torch::IntArrayRef batch_shape = {})
        : batch_shape_(batch_shape) {}
    Distribution(const std::vector<int64_t>& batch_shape = {})
        : batch_shape_(batch_shape) {}
    virtual ~Distribution() = default;

    virtual torch::Tensor sample(torch::IntArrayRef sample_shape) = 0;
    virtual torch::Tensor rsample(torch::IntArrayRef sample_shape) = 0;
    virtual torch::Tensor log_prob(const torch::Tensor& value) = 0;
    virtual torch::Tensor entropy() = 0;    
};

class Categorical : public Distribution 
{
    LAB_ARG(torch::Tensor, probs);
    LAB_ARG(torch::Tensor, logits);
    LAB_ARG(torch::Tensor, param);
public:
    Categorical(const torch::Tensor& in, bool is_logits = false);

    virtual torch::Tensor sample(torch::IntArrayRef sample_shape) override;
    virtual torch::Tensor rsample(torch::IntArrayRef sample_shape) override;
    virtual torch::Tensor log_prob(const torch::Tensor& value) override;
    virtual torch::Tensor entropy() override;
};

}
}