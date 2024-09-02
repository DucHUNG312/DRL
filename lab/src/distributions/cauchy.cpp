#include "lab/distributions/cauchy.h"

namespace lab
{
namespace distributions
{

Cauchy::Cauchy(const torch::Tensor& loc, const torch::Tensor& scale)
    : Distribution(loc.sizes())
{
    auto loc_scale = torch::broadcast_tensors({loc, scale});
    std::vector<int64_t> shape_vec = extended_shape();
    loc_ = loc_scale[0];
    scale_ = loc_scale[1];
    mean_ =     torch::full(
                torch::IntArrayRef(shape_vec),
                std::numeric_limits<double>::quiet_NaN(),
                torch::TensorOptions().dtype(torch::kDouble).device(loc_.device()));
    variance_ = torch::full(
                torch::IntArrayRef(shape_vec),
                std::numeric_limits<double>::max(),
                torch::TensorOptions().dtype(torch::kDouble).device(loc_.device()));
}

torch::Tensor Cauchy::rsample(torch::IntArrayRef sample_shape /*= {}*/)
{
    std::vector<int64_t> shape_vec = extended_shape(sample_shape);
    torch::IntArrayRef shape = torch::IntArrayRef(shape_vec);
    torch::Tensor eps = torch::empty(shape, loc_.options()).cauchy_();
    return loc_ + eps * scale_;
}

torch::Tensor Cauchy::log_prob(const torch::Tensor& value)
{
    return (
        - std::log(math::Pi)
        - scale_.log()
        - (((value - loc_) / scale_).pow(2)).log1p()
    );
}

torch::Tensor Cauchy::cdf(const torch::Tensor& value)
{
    return torch::atan((value - loc_) / scale_) / math::Pi + 0.5;
}

torch::Tensor Cauchy::icdf(const torch::Tensor& value)
{
    return torch::tan(math::Pi * (value - 0.5)) * scale_ + loc_;
}

torch::Tensor Cauchy::entropy()
{
    return std::log(4 * math::Pi) + scale_.log();
}

torch::Tensor Cauchy::sample(torch::IntArrayRef sample_shape /*= {}*/)
{
    LAB_UNIMPLEMENTED;
    return torch::Tensor();
}

torch::Tensor Cauchy::sample_n(int64_t n)
{
    LAB_UNIMPLEMENTED;
    return torch::Tensor();
}

}

}
