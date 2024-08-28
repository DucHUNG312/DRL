#include "lab/distributions/exp_family.h"

namespace lab
{
namespace distributions
{

torch::Tensor ExponentialFamily::log_normalizer(torch::TensorList params)
{
    LAB_UNIMPLEMENTED;
    return torch::Tensor();
}

torch::Tensor ExponentialFamily::entropy()
{
    torch::Tensor result = -mean_carrier_measure_;
    std::vector<torch::Tensor> nparams;
    for(const torch::Tensor& p : natural_params_)
        nparams.push_back(p.detach().clone().set_requires_grad(true));

    torch::Tensor lg_normal = log_normalizer(nparams);
    result += lg_normal;

    torch::TensorList gradients = torch::autograd::grad(
        {lg_normal.sum()}, 
        nparams, 
        /*grad_outputs=*/{}, 
        /*retain_graph=*/true, 
        /*create_graph=*/true
    );

    std::vector<int64_t> reshaped_shape(batch_shape_.begin(), batch_shape_.end());
    reshaped_shape.push_back(-1);
    for (size_t i = 0; i < nparams.size(); i++) 
        result -= (nparams[i] * gradients[i]).reshape(reshaped_shape).sum(-1);

    return result;
}
       
}
}