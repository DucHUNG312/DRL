#include "lab/utils/tensor.h"
#include "lab/utils/net.h"

namespace lab
{
namespace utils
{

bool has_no_zeros(const torch::Tensor& tensor)
{
    return torch::all(tensor).item<bool>();
}

bool has_all_zeros(const torch::Tensor& tensor)
{
    return tensor.nonzero().numel() == 0;
}

bool tensor_eq(const torch::Tensor& tensor1, const torch::Tensor& tensor2)
{
    torch::NoGradGuard no_grad;
    if(!tensor1.sizes().equals(tensor2.sizes())) return false;
    torch::Tensor mask = torch::eq(tensor1, tensor2);
    return mask.all().item<bool>();
}

bool tensor_lt(const torch::Tensor& tensor1, const torch::Tensor& tensor2)
{
    torch::NoGradGuard no_grad;
    if(!tensor1.sizes().equals(tensor2.sizes())) return false;
    torch::Tensor mask = torch::lt(tensor1, tensor2);
    return mask.all().item<bool>();
}

bool tensor_gt(const torch::Tensor& tensor1, const torch::Tensor& tensor2)
{
    torch::NoGradGuard no_grad;
    if(!tensor1.sizes().equals(tensor2.sizes())) return false;
    torch::Tensor mask = torch::gt(tensor1, tensor2);
    return mask.all().item<bool>();
}

bool tensor_le(const torch::Tensor& tensor1, const torch::Tensor& tensor2)
{
    torch::NoGradGuard no_grad;
    if(!tensor1.sizes().equals(tensor2.sizes())) return false;
    torch::Tensor mask = torch::le(tensor1, tensor2);
    return mask.all().item<bool>();
}

bool tensor_ge(const torch::Tensor& tensor1, const torch::Tensor& tensor2)
{
    torch::NoGradGuard no_grad;
    if(!tensor1.sizes().equals(tensor2.sizes())) return false;
    torch::Tensor mask = torch::ge(tensor1, tensor2);
    return mask.all().item<bool>();
}

bool tensor_close(const torch::Tensor& tensor1, const torch::Tensor& tensor2)
{
    torch::NoGradGuard no_grad;
    if(!tensor1.sizes().equals(tensor2.sizes())) return false;
    torch::Tensor mask = torch::isclose(tensor1, tensor2);
    return mask.all().item<bool>();
}

std::string get_object_name(const c10::IValue& ivalue)
{
    if (ivalue.isObject())
    {
        auto obj_type = ivalue.toObjectRef().type();
        return obj_type->name()->name();
    }
    LAB_UNREACHABLE;
    return "";
}

torch::Tensor center_mean(const torch::Tensor& tensor)
{
    torch::Tensor ret = tensor - torch::mean(tensor);
    return ret;
}

torch::Tensor center_mean(const std::vector<double>& vec) 
{
    double mean = std::accumulate(vec.begin(), vec.end(), 0.0) / vec.size();
    std::vector<double> centered_vec(vec.size());
    std::transform(vec.begin(), vec.end(), centered_vec.begin(), [mean](double val) { 
        return val - mean; 
    });
    torch::Tensor centered_tensor = torch::tensor(centered_vec);
    return centered_tensor;
}

torch::Tensor normalize(const torch::Tensor& tensor)
{
    auto range = tensor.max() - tensor.min();
    range += 1e-08;  // division guard
    torch::Tensor norm = (tensor - tensor.min()) / range;
    return norm;
}

torch::Tensor standardize(const torch::Tensor& tensor)
{
    LAB_CHECK(tensor.numel() > 1);
    torch::Tensor std = (tensor - torch::mean(tensor)) / (torch::std(tensor) + + 1e-08);
    return std;
}

torch::Tensor to_one_hot(const torch::Tensor& tensor, int64_t num_classes) 
{
    LAB_CHECK((tensor.dtype() == torch::kInt64) && (tensor.dim() == 1));    
    auto one_hot_tensor = torch::zeros(tensor.sizes().vec(), torch::kInt64)
                            .unsqueeze(-1).expand({-1, -1, num_classes})
                            .scatter_(-1, tensor.unsqueeze(-1), 1);
    return one_hot_tensor;
}

torch::Tensor venv_pack(const torch::Tensor& batch_tensor, int64_t num_envs)
{
    torch::IntArrayRef shape = batch_tensor.sizes();
    if(shape.size() < 2)
        return batch_tensor.view({-1, num_envs});
   
    std::vector<int64_t> pack_shape = {-1, num_envs};
    pack_shape.insert(pack_shape.end(), shape.begin() + 1, shape.end());
    return batch_tensor.view(pack_shape);
}

torch::Tensor venv_unpack(const torch::Tensor& batch_tensor)
{
    torch::IntArrayRef shape = batch_tensor.sizes();
    if (shape.size() < 3)
        return batch_tensor.view(-1);
   
    std::vector<int64_t> unpack_shape = {-1};
    unpack_shape.insert(unpack_shape.end(), shape.begin() + 2, shape.end());
    return batch_tensor.view(unpack_shape);
}

torch::Tensor calc_q_value_logits(const torch::Tensor& state_value, const torch::Tensor& raw_advantages)
{
    torch::Tensor mean_adv = raw_advantages.mean(-1).unsqueeze(-1);
    return (state_value + raw_advantages - mean_adv);
}

}
}