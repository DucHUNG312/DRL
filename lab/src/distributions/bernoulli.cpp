#include "lab/distributions/bernoulli.h"
#include "lab/distributions/utils.h"

namespace lab
{
namespace distributions
{

Bernoulli::Bernoulli(const torch::Tensor& in, bool is_logits /*= false*/)
    : ExponentialFamily(in.sizes()), is_logits_(is_logits)
{  
    params_ = in;
    mean_ = probs();
    variance_ = probs() * (1 - probs());
    natural_params_ = torch::TensorList({torch::logit(probs())});
}

torch::Tensor Bernoulli::logits()
{
    return is_logits_ ? params_ : probs_to_logits(params_);
}

torch::Tensor Bernoulli::probs()
{
    return is_logits_ ? logits_to_probs(params_) : params_;
}

torch::IntArrayRef Bernoulli::params_shape()
{
    return params_.sizes();
}

torch::Tensor Bernoulli::sample(torch::IntArrayRef sample_shape /*= {}*/)
{
    torch::NoGradGuard no_grad;
    std::vector<int64_t> shape_vec = extended_shape(sample_shape);
    torch::IntArrayRef shape = torch::IntArrayRef(shape_vec);
    return torch::bernoulli(probs().expand(shape));
}

torch::Tensor Bernoulli::log_prob(const torch::Tensor& value)
{
    auto logits_value = torch::broadcast_tensors({logits(), value});
    torch::Tensor logits = logits_value[0];
    torch::Tensor val = logits_value[1];
    return -torch::nn::functional::binary_cross_entropy_with_logits(logits, val);
}

torch::Tensor Bernoulli::entropy()
{
    return torch::nn::functional::binary_cross_entropy_with_logits(logits(), probs());
}

torch::Tensor Bernoulli::log_normalizer(torch::TensorList params)
{
    LAB_CHECK_EQ(params.size(), 1);
    return torch::log1p(torch::exp(params[0]));
}

}

}
