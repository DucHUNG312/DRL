#pragma once

#include "lab/core.h"

namespace lab
{
namespace utils
{
bool str_to_bool(const std::string& str);

#if 0
torch::Tensor cv_to_tensor(const cv::Mat& in_mat, int64_t h, int64_t w, int64_t c);

cv::Mat tensor_to_cv(const torch::Tensor& in_tensor, size_t h, size_t w, size_t c);

torch::Tensor eigen_to_tensor(Mat& in_mat);

Mat eigen_to_tensor(const torch::Tensor& in_tensor);
#endif

std::vector<double> get_data_from_tensor(const torch::Tensor& tensor);

torch::Tensor get_tensor_from_vec(const std::vector<double>& vec, const std::vector<int64_t>& shape);

torch::Tensor get_tensor_from_ivalue_list(const torch::List<torch::IValue>& list);

std::vector<double> get_rewards_from_ivalue_list(const torch::List<torch::IValue>& list);

std::vector<bool> get_dones_from_ivalue_list(const torch::List<torch::IValue>& list);

template<typename IteratorIn, typename IteratorOut>
void void_to_string(IteratorIn first, IteratorIn last, IteratorOut out)
{
    std::transform(first, last, out, [](auto d) { return std::to_string(d); } );
}

template<typename IteratorIn, typename IteratorOut>
void string_vec_to_const_char(IteratorIn first, IteratorIn last, IteratorOut out)
{
    std::transform(first, last, out, [](const std::string& s) { return s.c_str(); } );
}

template<typename IteratorIn, typename IteratorOut>
void void_to_const_char(IteratorIn first, IteratorIn last, IteratorOut out)
{
    std::vector<std::string> temp;
    temp.reserve(std::distance(first, last));
    void_to_string(first, last, std::back_inserter(temp));
    string_vec_to_const_char(temp.begin(), temp.end(), out);
}

}
}