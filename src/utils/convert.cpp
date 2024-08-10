#include "lab/utils/convert.h"

namespace lab
{
namespace utils
{
torch::Tensor cv_to_tensor(const cv::Mat& in_mat, size_t h, size_t w, size_t c) 
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
}
}