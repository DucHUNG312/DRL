#include "lab/distributions/dirichlet.h"

namespace lab
{
namespace distributions
{

/// class MyFunction : public Function<MyFunction> {
///   public:
///   static constexpr bool is_traceable = true;
///
///   static variable_list forward(AutogradContext *ctx, int n, Variable var) {
///      // Save data for backward in context
///      ctx->saved_data["n"] = n;
///      var.mul_(2);
///      // Mark var as modified by inplace operation
///      ctx->mark_dirty({var});
///      return {var};
///   }
///
///   static variable_list backward(AutogradContext *ctx, variable_list
///   grad_output) {
///      // Use data saved in forward
///      auto n = ctx->saved_data["n"].toInt();
///      return {grad_output[0]*n};
///   }
/// };
/// ```
///
/// To use `MyFunction`:
/// ```
/// Variable x;
/// auto y = MyFunction::apply(6, x);
/// // Example backward call
/// y[0].sum().backward();

torch::Tensor dirichlet_backward_(const torch::Tensor& x, const torch::Tensor& concentration, const torch::Tensor& grad_output) 
{
    torch::Tensor total = concentration.sum(-1, true).expand_as(concentration);
    torch::Tensor grad = at::_dirichlet_grad(x, concentration, total);
    return grad * (grad_output - (x * grad_output).sum(-1, true));
}

torch::Tensor Dirichlet_::forward(torch::autograd::AutogradContext* ctx, const torch::Tensor& concentration) 
{
    torch::Tensor x = torch::_sample_dirichlet(concentration);
    ctx->save_for_backward({x, concentration});
    return x;
}

torch::autograd::variable_list Dirichlet_::backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_output) 
{
    std::vector<torch::Tensor> saved = ctx->get_saved_variables();
    torch::Tensor x = saved[0];
    torch::Tensor concentration = saved[1];
    auto grad_output_tensor = grad_output[0];
    torch::Tensor grad_input = dirichlet_backward_(x, concentration, grad_output_tensor);
    return { grad_input };
}

Dirichlet::Dirichlet(const torch::Tensor& concentration)
    : ExponentialFamily(concentration.sizes().slice(0, concentration.dim() - 1)), 
        concentration_(concentration)
{
    LAB_CHECK(concentration.dim() >= 1);
    mean_ = concentration / (concentration.sum(-1, true));
    torch::Tensor con0 = concentration.sum(-1, true);
    variance_ = concentration * (con0 - concentration) / (con0.pow(2) * (con0 + 1));
    natural_params_ = torch::TensorList(concentration);
}

Dirichlet::Dirichlet(const torch::Tensor& concentration, const torch::Tensor&)
    : Dirichlet(concentration) {};

torch::Tensor Dirichlet::rsample(torch::IntArrayRef sample_shape /*= {}*/)
{
    torch::IntArrayRef shape = extended_shape(sample_shape);
    torch::Tensor concentration = concentration_.expand(shape);
    return Dirichlet_::apply(concentration_);
}

torch::Tensor Dirichlet::log_prob(const torch::Tensor& value)
{
    return torch::xlogy(concentration_ - 1.0, value).sum(-1)
        +  torch::lgamma(concentration_.sum(-1))
        -  torch::lgamma(concentration_).sum(-1);
}

torch::Tensor Dirichlet::entropy()
{
    int64_t k = concentration_.size(-1);
    torch::Tensor a0 = concentration_.sum(-1);
    return (
        torch::lgamma(concentration_).sum(-1)
        - torch::lgamma(a0)
        - (k - a0) * torch::digamma(a0)
        - ((concentration_ - 1.0) * torch::digamma(concentration_)).sum(-1)
    );
}

torch::Tensor Dirichlet::log_normalizer(torch::TensorList params)
{
    LAB_CHECK_EQ(params.size(), 1);
    return params[0].lgamma().sum(-1) - torch::lgamma(params[0].sum(-1));
}

}

}