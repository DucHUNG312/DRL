#include "lab/utils/tensor.h"

namespace lab
{
namespace utils
{
namespace internal
{
template<typename T>
T get_bounding_shape(const std::vector<T>& array_refs)
{
    size_t max_dimensions = 0;
    for (const auto& array : array_refs)
        max_dimensions = std::max(max_dimensions, array.size());

    std::vector<int64_t> bounding_shape(max_dimensions, 0);

    for (const auto& array : array_refs)
        for (size_t i = 0; i < array.size(); i++)
            bounding_shape[i] = std::max(bounding_shape[i], array[i]);

    return T(bounding_shape);
}
}

bool has_no_zeros(const torch::Tensor& tensor)
{
    return torch::all(tensor).item<bool>();
}

bool has_all_zeros(const torch::Tensor& tensor)
{
    return tensor.nonzero().numel() == 0;
}

torch::IntArrayRef get_bounding_shape(const std::vector<torch::IntArrayRef>& array_refs) 
{
    return internal::get_bounding_shape<torch::IntArrayRef>(array_refs);
}

IShape get_bounding_shape(const std::vector<IShape>& array_refs)
{
    return internal::get_bounding_shape<IShape>(array_refs);
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

torch::Tensor clamp_probs(const torch::Tensor& probs)
{
    auto eps = (probs.dtype() == torch::kDouble) ? std::numeric_limits<double>::epsilon() : std::numeric_limits<float>::epsilon();
    return torch::clamp(probs, eps, 1. - eps);
} 

torch::Tensor probs_to_logits(const torch::Tensor& probs, bool is_binary /*= false*/)
{
    torch::Tensor ps_clamped = clamp_probs(probs);
    if(is_binary) return torch::log(ps_clamped) - torch::log1p(-ps_clamped);
    return torch::log(ps_clamped);
}

torch::Tensor logits_to_probs(const torch::Tensor& logits, bool is_binary /*= false*/)
{
    if(is_binary) return torch::sigmoid(logits);
    return torch::nn::functional::softmax(logits, {-1});
}

}
}