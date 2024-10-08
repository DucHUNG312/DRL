#include "lab/utils/convert.h"

namespace lab {
namespace utils {
bool str_to_bool(const std::string& str) {
  if (str == "0")
    return false;
  else if (str == "1")
    return true;
  LAB_UNREACHABLE;
  return false;
}

#if 0
torch::Tensor cv_to_tensor(const cv::Mat& in_mat, int64_t h, int64_t w, int64_t c) 
{
    cv::Mat mat = in_mat.clone();
    cv::resize(mat, mat, cv::Size(w, h));
    if (c == 1)
        mat.convertTo(mat, CV_32FC1, 1.0f / 255.0f);
    else if(c == 3) 
    {
        mat.convertTo(mat, CV_32FC3, 1.0f / 255.0f);
        cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
    }
    auto tensor = torch::from_blob(mat.data, {h, w, c}).permute({2, 0, 1});
    return tensor.clone();
}

cv::Mat tensor_to_cv(const torch::Tensor& in_tensor, size_t h, size_t w, size_t c)
{
    torch::Tensor tensor = in_tensor.clone();
    tensor = tensor.detach().permute({1, 2, 0}).contiguous();
    tensor = tensor.mul(255).clamp(0, 255).to(torch::kU8);
    tensor = tensor.to(torch::kCPU);
    cv::Mat mat;
    if (c == 1)
    {
        mat = cv::Mat(h, w, CV_8UC1);
        std::memcpy((void *) mat.data, tensor.data_ptr(), sizeof(torch::kU8) * tensor.numel());
    }
    else if (c == 3)
    {
        mat = cv::Mat(h, w, CV_8UC3);
        std::memcpy((void *) mat.data, tensor.data_ptr(), sizeof(torch::kU8) * tensor.numel());
        cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
    }
    return mat;
}

torch::Tensor eigen_to_tensor(Mat& in_mat)
{
    auto tensor = torch::from_blob(in_mat.data(), {in_mat.rows(), in_mat.cols()}).to(torch::kDouble).to(torch::kCPU);
    return tensor.clone();
}

Mat eigen_to_tensor(const torch::Tensor& in_tensor)
{
    auto tensor = in_tensor.to(torch::kDouble).to(torch::kCPU);
    Eigen::Map<Mat> mat(tensor.data_ptr<double>(), tensor.size(0), tensor.size(1));
    return mat;
}
#endif

std::vector<double> get_data_from_tensor(const torch::Tensor& tensor) {
  torch::Tensor tensor_ = tensor.to(torch::kDouble).view(-1);
  return std::vector<double>(tensor_.data_ptr<double>(), tensor_.data_ptr<double>() + tensor_.numel());
}

torch::Tensor get_tensor_from_vec(const std::vector<double>& vec, const std::vector<int64_t>& shape) {
  int64_t total_size = 1;
  for (int64_t dim : shape)
    total_size *= dim;

  if (vec.size() != total_size)
    throw std::invalid_argument("Size of vector does not match the specified shape.");

  auto options = torch::TensorOptions().dtype(torch::kDouble).requires_grad(false);
  torch::Tensor tensor = torch::empty(shape, options);
  std::memcpy(tensor.data_ptr(), vec.data(), vec.size() * sizeof(double));

  return tensor;
}

torch::Tensor get_tensor_from_ivalue_list(const torch::List<torch::IValue>& list) {
  std::vector<torch::Tensor> tensor_vec;
  tensor_vec.reserve(list.size());
  for (const auto& ele : list.vec())
    tensor_vec.push_back(ele.toTensor());
  return torch::stack(tensor_vec).to(torch::kDouble);
}

std::vector<double> get_rewards_from_ivalue_list(const torch::List<torch::IValue>& list) {
  std::vector<double> double_vec;
  double_vec.reserve(list.size());
  for (const auto& ele : list.vec())
    double_vec.push_back(ele.toDouble());
  return double_vec;
}

std::vector<bool> get_dones_from_ivalue_list(const torch::List<torch::IValue>& list) {
  std::vector<bool> bool_vec;
  bool_vec.reserve(list.size());
  for (const auto& ele : list.vec())
    bool_vec.push_back(ele.toBool());
  return bool_vec;
}

} // namespace utils
} // namespace lab