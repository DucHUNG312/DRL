#include "lab/utils/distribution.h"
#include "lab/utils/tensor.h"

namespace lab
{
namespace utils
{

Categorical::Categorical(const torch::Tensor& in, bool is_logits /*= false*/) 
    : Distribution(in.sizes())
{
    if(is_logits)
    {
        logits_ = in - in.logsumexp({-1}, true);
        probs_ = utils::logits_to_probs(logits_);
    }
    else
    {
        probs_ = in / in.sum({-1}, true);
        logits_ = utils::probs_to_logits(probs_);
    }
    
    param_ = is_logits ? logits_ : probs_;
}

torch::Tensor Categorical::sample(const utils::IShape& sample_shape)
{
    LAB_UNIMPLEMENTED;
    return torch::Tensor();
}

torch::Tensor Categorical::rsample(const utils::IShape& sample_shape)
{
    return sample(sample_shape);
}

torch::Tensor Categorical::log_prob(const torch::Tensor& value)
{
    LAB_UNIMPLEMENTED;
    return torch::Tensor();
}

torch::Tensor Categorical::entropy()
{
    LAB_UNIMPLEMENTED;
    return torch::Tensor();
}   

}
}