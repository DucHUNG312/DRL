#pragma once

#include "lab/core.h"

namespace lab
{
namespace utils
{
torch::Tensor cv_to_tensor(const cv::Mat& in_mat, size_t h, size_t w, size_t c);

cv::Mat tensor_to_cv(const torch::Tensor& in_tensor, size_t h, size_t w, size_t c);
}
}