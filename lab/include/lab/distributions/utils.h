#pragma once

#include "lab/common/common.h"

namespace lab {
namespace distributions {

torch::Tensor clamp_probs(const torch::Tensor& probs, const std::optional<torch::Device>& device = std::nullopt);

torch::Tensor probs_to_logits(
    const torch::Tensor& probs,
    bool is_binary = false,
    const std::optional<torch::Device>& device = std::nullopt);

torch::Tensor logits_to_probs(
    const torch::Tensor& logits,
    bool is_binary = false,
    const std::optional<torch::Device>& device = std::nullopt);

torch::Tensor standard_normal(const torch::IntArrayRef& shape, const torch::Dtype& dtype, const torch::Device& device);

} // namespace distributions

} // namespace lab