#pragma once

#include "lab/core.h"

namespace lab
{
namespace utils
{
bool str_to_bool(const std::string& str);

torch::Tensor cv_to_tensor(const cv::Mat& in_mat, int64_t h, int64_t w, int64_t c);

cv::Mat tensor_to_cv(const torch::Tensor& in_tensor, size_t h, size_t w, size_t c);

torch::Tensor eigen_to_tensor(Mat& in_mat);

Mat eigen_to_tensor(const torch::Tensor& in_tensor);

std::vector<int64_t> get_arayref_data(torch::IntArrayRef arr);

}
}