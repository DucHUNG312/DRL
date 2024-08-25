#include "lab/spaces/box.h"
#include "lab/utils/tensor.h"

namespace lab
{
namespace spaces
{

BoxOptions::BoxOptions(const torch::Tensor& low, const torch::Tensor& high)
{
    low_ = low.to(torch::kDouble);
    high_ = high.to(torch::kDouble);
}

BoxImpl::BoxImpl(const BoxOptions& options_)
    : options(options_)
{
    reset();
}

void BoxImpl::reset()
{
    low = register_parameter("low", options.low());
    high = register_parameter("high", options.high());
    LAB_CHECK_EQ(low.sizes(), high.sizes());
    shape_ = torch::tensor(low.sizes().vec(), torch::kInt64);
    name_ = "Box";
}

torch::Tensor BoxImpl::sample()
{
    torch::Tensor sample = rand_.sample_uniform(low, high);
    return sample.clone();
}

bool BoxImpl::contains(const torch::Tensor& x) const
{
    return (utils::tensor_ge(x, low) && utils::tensor_le(x, high));
}

}
}