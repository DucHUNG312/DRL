#include "lab/distributions/beta.h"

namespace lab
{
namespace distributions
{

Beta::Beta(const torch::Tensor& concentration0, const torch::Tensor& concentration1)
{
    auto concentrations = torch::broadcast_tensors({concentration1, concentration0});
    torch::Tensor concen1 = concentrations[0];
    torch::Tensor concen0 = concentrations[1];
    torch::Tensor concentration1_concentration0 = torch::stack({concen1, concen0}, -1);
    dirichlet_ = Dirichlet(concentration1_concentration0);

    concentration1_ = dirichlet_.concentration().index({torch::indexing::Ellipsis, 0});
    concentration0_ = dirichlet_.concentration().index({torch::indexing::Ellipsis, 1});

    natural_params_ = torch::TensorList({concentration1_, concentration0_});

    batch_shape_ = dirichlet_.batch_shape();
    mean_ = concentration1_ / (concentration1_ + concentration0_);
    variance_ = (concentration1_ * concentration0_) / ((concentration1_ + concentration0_).pow(2) * (concentration1_ + concentration0_ + 1));
}

torch::Tensor Beta::rsample(torch::IntArrayRef sample_shape)
{
    return dirichlet_.rsample(sample_shape).select(-1, 0);
}

torch::Tensor Beta::log_prob(const torch::Tensor& value)
{
    torch::Tensor heads_tails = torch::stack({value, 1.0 - value}, -1);
    return dirichlet_.log_prob(heads_tails);
}

torch::Tensor Beta::log_normalizer(torch::TensorList params)
{
    LAB_CHECK_EQ(params.size(), 2);
    return torch::lgamma(params[0]) + torch::lgamma(params[1]) - torch::lgamma(params[0] + params[1]);
}

}

}