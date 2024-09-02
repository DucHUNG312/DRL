#pragma once

#include "lab/core.h"

namespace lab
{
namespace utils
{

bool has_no_zeros(const torch::Tensor& tensor);

bool has_all_zeros(const torch::Tensor& tensor);

bool tensor_eq(const torch::Tensor& tensor1, const torch::Tensor& tensor2);

bool tensor_lt(const torch::Tensor& tensor1, const torch::Tensor& tensor2);

bool tensor_gt(const torch::Tensor& tensor1, const torch::Tensor& tensor2);

bool tensor_le(const torch::Tensor& tensor1, const torch::Tensor& tensor2);

bool tensor_ge(const torch::Tensor& tensor1, const torch::Tensor& tensor2);

bool tensor_close(const torch::Tensor& tensor1, const torch::Tensor& tensor2);

std::string get_object_name(const c10::IValue& ivalue);

torch::Tensor center_mean(const torch::Tensor& tensor);

torch::Tensor center_mean(const std::vector<double>& vec);

torch::Tensor normalize(const torch::Tensor& tensor);

torch::Tensor standardize(const torch::Tensor& tensor);

torch::Tensor to_one_hot(const torch::Tensor& tensor, int64_t num_classes);

torch::Tensor venv_pack(const torch::Tensor& batch_tensor, int64_t num_envs);

torch::Tensor venv_unpack(const torch::Tensor& batch_tensor);

torch::Tensor calc_q_value_logits(const torch::Tensor& state_value, const torch::Tensor& raw_advantages);
}
}