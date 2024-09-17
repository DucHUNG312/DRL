#include "lab/distributions/utils.h"
#include <torch/csrc/jit/frontend/tracer.h>

namespace lab {
namespace distributions {

torch::Tensor clamp_probs(const torch::Tensor& probs, const std::optional<torch::Device>& device /*= std::nullopt*/) {
  auto eps = (probs.dtype() == torch::kDouble) ? std::numeric_limits<double>::epsilon()
                                               : std::numeric_limits<float>::epsilon();
  return device.has_value() ? torch::clamp(probs, eps, 1. - eps).to(device.value())
                            : torch::clamp(probs, eps, 1. - eps);
}

torch::Tensor probs_to_logits(
    const torch::Tensor& probs,
    bool is_binary /*= false*/,
    const std::optional<torch::Device>& device /*= std::nullopt*/) {
  torch::Tensor ps_clamped = clamp_probs(probs);
  if (is_binary)
    return torch::log(ps_clamped) - torch::log1p(-ps_clamped);
  return device.has_value() ? torch::log(ps_clamped).to(device.value()) : torch::log(ps_clamped);
}

torch::Tensor logits_to_probs(
    const torch::Tensor& logits,
    bool is_binary /*= false*/,
    const std::optional<torch::Device>& device /*= std::nullopt*/) {
  if (is_binary)
    return device.has_value() ? torch::sigmoid(logits).to(device.value()) : torch::sigmoid(logits);
  return device.has_value() ? torch::nn::functional::softmax(logits, {-1}).to(device.value())
                            : torch::nn::functional::softmax(logits, {-1});
}

torch::Tensor standard_normal(const torch::IntArrayRef& shape, const torch::Dtype& dtype, const torch::Device& device) {
  if (torch::jit::tracer::isTracing()) {
    return at::normal(
        torch::zeros(shape, torch::TensorOptions().dtype(dtype).device(device)),
        torch::ones(shape, torch::TensorOptions().dtype(dtype).device(device)));
  }

  return torch::empty(shape, torch::TensorOptions().dtype(dtype).device(device)).normal_();
}

} // namespace distributions

} // namespace lab