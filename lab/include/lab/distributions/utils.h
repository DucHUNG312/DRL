#pragma once

#include "lab/core.h"

namespace lab
{
namespace distributions
{

torch::Tensor clamp_probs(const torch::Tensor& probs);

torch::Tensor probs_to_logits(const torch::Tensor& probs, bool is_binary = false);

torch::Tensor logits_to_probs(const torch::Tensor& logits, bool is_binary = false);

torch::Tensor standard_normal(const torch::IntArrayRef& shape, const torch::Dtype& dtype, const torch::Device& device);


}

}