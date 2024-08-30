#pragma once

#include "lab/distributions/exp_family.h"
#include <ATen/ATen.h>

namespace lab
{
namespace distributions
{

torch::Tensor dirichlet_backward_(const torch::Tensor& x, const torch::Tensor& concentration, const torch::Tensor& grad_output);

class Dirichlet_ : public torch::autograd::Function<Dirichlet_>
{
public:
    static constexpr bool is_traceable = true;
    
    static torch::Tensor forward(torch::autograd::AutogradContext* ctx, const torch::Tensor& concentration);

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_output);
};

class Dirichlet : public ExponentialFamily
{
    LAB_ARG(torch::Tensor, concentration);
public:
    Dirichlet(const torch::Tensor& concentration);
    Dirichlet(const torch::Tensor& concentration, const torch::Tensor&); // this constructor only used to satisfy the construction condition in ContinuousActionPDFactory
    LAB_DEFAULT_CONSTRUCT(Dirichlet);

    torch::Tensor rsample(torch::IntArrayRef sample_shape = {}) override;

    torch::Tensor log_prob(const torch::Tensor& value) override;
    
    torch::Tensor entropy() override;

    torch::Tensor log_normalizer(torch::TensorList params) override;
};

}

}