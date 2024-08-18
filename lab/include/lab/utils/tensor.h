#pragma once

#include "lab/core.h"
#include "lab/utils/shape.h"

namespace lab
{
namespace utils
{
/**
 * @brief Check if the tensor contains any zero elements
 * @param tensor tensor to check
 */
bool has_no_zeros(const torch::Tensor& tensor);

/**
 * @brief Check if the tensor contains all zero elements
 * @param tensor tensor to check
 */
bool has_all_zeros(const torch::Tensor& tensor);

/**
 * @brief Get bounding shape for a list of shape
 * @param array_refs list of shapes
 */
torch::IntArrayRef get_bounding_shape(const std::vector<torch::IntArrayRef>& array_refs);

/**
 * @brief Get bounding shape for a list of shape int
 * @param array_refs list of shapes int
 */
IShape get_bounding_shape(const std::vector<IShape>& array_refs);

/**
 * @brief Check if 2 tensor are equal
 * @param tensor1 tensor1
 * @param tensor2 tensor2
 */
bool tensor_eq(const torch::Tensor& tensor1, const torch::Tensor& tensor2);

/**
 * @brief Check if tensor1 is less than tensor 2
 * @param tensor1 tensor1
 * @param tensor2 tensor2
 */
bool tensor_lt(const torch::Tensor& tensor1, const torch::Tensor& tensor2);

/**
 * @brief Check if tensor1 is greater than tensor 2
 * @param tensor1 tensor1
 * @param tensor2 tensor2
 */
bool tensor_gt(const torch::Tensor& tensor1, const torch::Tensor& tensor2);

/**
 * @brief Check if tensor1 is less than or equal tensor 2
 * @param tensor1 tensor1
 * @param tensor2 tensor2
 */
bool tensor_le(const torch::Tensor& tensor1, const torch::Tensor& tensor2);

/**
 * @brief Check if tensor1 is greater than or equal tensor 2
 * @param tensor1 tensor1
 * @param tensor2 tensor2
 */
bool tensor_ge(const torch::Tensor& tensor1, const torch::Tensor& tensor2);

/**
 * @brief Check if tensor1 is close to tensor 2
 * @param tensor1 tensor1
 * @param tensor2 tensor2
 */
bool tensor_close(const torch::Tensor& tensor1, const torch::Tensor& tensor2);

/**
 * @brief Get the object name
 * @param ivalue object
 */
std::string get_object_name(const c10::IValue& ivalue);

torch::Tensor clamp_probs(const torch::Tensor& probs);

torch::Tensor probs_to_logits(const torch::Tensor& probs, bool is_binary = false);

torch::Tensor logits_to_probs(const torch::Tensor& logits, bool is_binary = false);
}
}