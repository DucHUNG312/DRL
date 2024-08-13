#pragma once

#include "lab/core.h"

namespace lab
{
namespace utils
{
bool str_to_bool(const std::string& str);

torch::Tensor cv_to_tensor(const cv::Mat& in_mat, size_t h, size_t w, size_t c);

cv::Mat tensor_to_cv(const torch::Tensor& in_tensor, size_t h, size_t w, size_t c);

/**
 * Retun a double tensor
 */
torch::Tensor eigen_to_tensor(Mat& in_mat);

Mat eigen_to_tensor(const torch::Tensor& in_tensor);

}
}