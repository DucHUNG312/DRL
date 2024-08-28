#include "lab/distributions/utils.h"
#include <torch/csrc/jit/frontend/tracer.h>

namespace lab
{
namespace distributions
{

torch::Tensor clamp_probs(const torch::Tensor& probs)
{
    auto eps = (probs.dtype() == torch::kDouble) ? std::numeric_limits<double>::epsilon() : std::numeric_limits<float>::epsilon();
    return torch::clamp(probs, eps, 1. - eps).to(utils::get_torch_device());
} 

torch::Tensor probs_to_logits(const torch::Tensor& probs, bool is_binary /*= false*/)
{
    torch::Tensor ps_clamped = clamp_probs(probs);
    if(is_binary) return torch::log(ps_clamped) - torch::log1p(-ps_clamped);
    return torch::log(ps_clamped).to(utils::get_torch_device());
}

torch::Tensor logits_to_probs(const torch::Tensor& logits, bool is_binary /*= false*/)
{
    if(is_binary) return torch::sigmoid(logits);
    return torch::nn::functional::softmax(logits, {-1}).to(utils::get_torch_device());
}

torch::Tensor standard_normal(const torch::IntArrayRef& shape, const torch::Dtype& dtype, const torch::Device& device) 
{
    if (torch::jit::tracer::isTracing()) 
    {
        return at::normal(
            torch::zeros(shape, torch::TensorOptions().dtype(dtype).device(device)),
            torch::ones(shape, torch::TensorOptions().dtype(dtype).device(device))
        );
    }

    return torch::empty(shape, torch::TensorOptions().dtype(dtype).device(device)).normal_();
}

}

}