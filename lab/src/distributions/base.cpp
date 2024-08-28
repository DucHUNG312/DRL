#include "lab/distributions/base.h"
#include "lab/utils/tensor.h"

namespace lab
{
namespace distributions
{

Distribution::Distribution(torch::IntArrayRef batch_shape)
    : batch_shape_(batch_shape) {}

torch::IntArrayRef Distribution::extended_shape(torch::IntArrayRef sample_shape /*= {}*/)
{
    std::vector<int64_t> combined_shape;
    if(!sample_shape.empty())
        combined_shape.insert(combined_shape.end(), sample_shape.begin(), sample_shape.end());
    combined_shape.insert(combined_shape.end(), batch_shape_.begin(), batch_shape_.end());
    return torch::IntArrayRef(combined_shape);
}

torch::Tensor Distribution::sample(torch::IntArrayRef sample_shape)
{
    LAB_UNIMPLEMENTED;
    return torch::Tensor();
}

torch::Tensor Distribution::rsample(torch::IntArrayRef sample_shape)
{
    LAB_UNIMPLEMENTED;
    return torch::Tensor();
}

torch::Tensor Distribution::sample_n(int64_t n)
{
    LAB_UNIMPLEMENTED;
    return torch::Tensor();
}

torch::Tensor Distribution::log_prob(const torch::Tensor& value)
{
    LAB_UNIMPLEMENTED;
    return torch::Tensor();
}

torch::Tensor Distribution::cdf(const torch::Tensor& value)
{
    LAB_UNIMPLEMENTED;
    return torch::Tensor();
}

torch::Tensor Distribution::icdf(const torch::Tensor& value)
{
    LAB_UNIMPLEMENTED;
    return torch::Tensor();
}   

torch::Tensor Distribution::entropy()
{
    LAB_UNIMPLEMENTED;
    return torch::Tensor();
}

}
}