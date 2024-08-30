#include "lab/distributions/normal.h"
#include "lab/distributions/utils.h"

namespace lab
{
namespace distributions
{

Normal::Normal(const torch::Tensor& loc, const torch::Tensor& scale)
    : ExponentialFamily(loc.sizes())
{
    auto loc_scale = torch::broadcast_tensors({loc, scale});
    loc_ = loc_scale[0];
    scale_ = loc_scale[1];
    mean_ = loc_;
    stddev_ = scale_;
    variance_ = stddev_.pow(2);
    natural_params_ = torch::TensorList({loc_ / scale_.pow(2), -0.5 * scale_.pow(2).reciprocal()});
}

torch::Tensor Normal::sample(torch::IntArrayRef sample_shape /*= {}*/)
{
    torch::NoGradGuard no_grad;
    torch::IntArrayRef shape = extended_shape(sample_shape);
    return at::normal(loc_.expand(shape), scale_.expand(shape));
}

torch::Tensor Normal::rsample(torch::IntArrayRef sample_shape /*= {}*/)
{
    torch::IntArrayRef shape = extended_shape(sample_shape);
    torch::Tensor eps = standard_normal(shape, loc_.dtype().toScalarType(), loc_.device());
    return loc_ + eps * scale_;
}

torch::Tensor Normal::log_prob(const torch::Tensor& value)
{
    return (
        -((value - loc_).pow(2)) / (2 * variance_)
        - scale_.log()
        - std::log(std::sqrt(2 * math::Pi))
    );
}

torch::Tensor Normal::cdf(const torch::Tensor& value)
{
    return 0.5 * (1 + torch::erf((value - loc_) * scale_.reciprocal() / std::sqrt(2)));
}

torch::Tensor Normal::icdf(const torch::Tensor& value)
{
    return loc_ + scale_ * torch::erfinv(2 * value - 1) * std::sqrt(2);
} 

torch::Tensor Normal::entropy()
{
    return 0.5 + 0.5 * std::log(2 * math::Pi) + torch::log(scale_);
}

torch::Tensor Normal::log_normalizer(torch::TensorList params)
{
    LAB_CHECK_EQ(params.size(), 2);
    return  -0.25 * params[0].pow(2) / params[1] + 0.5 * torch::log(math::Pi / params[1]);
}

}

}