#pragma once

#include "lab/core.h"
#include "lab/utils/typetraits.h"

namespace lab
{

namespace utils
{
struct ActivationModule : public torch::nn::Module
{
    using torch::nn::Module::Module;
    virtual torch::Tensor forward(const torch::Tensor& input) = 0;
};

struct LossModule : public torch::nn::Module
{
    using torch::nn::Module::Module;
    virtual torch::Tensor forward(const torch::Tensor& input, const torch::Tensor& target) = 0;
};

class NoGradGuard
{
public:
    NoGradGuard();
    virtual ~NoGradGuard();
private:
    std::unique_ptr<torch::NoGradGuard> no_grad_guard;
};

#define LAB_ACTIVATION_DECLARE(Name)                                                                    \
struct Name : public torch::nn::Name##Impl, public lab::utils::ActivationModule                         \
{                                                                                                       \
    using torch::nn::Name##Impl::Name##Impl;                                                            \
    static constexpr const char* name = #Name;                                                          \
    torch::Tensor forward(const torch::Tensor& input) override                                          \
    {                                                                                                   \
        return torch::nn::Name##Impl::forward(input);                                                   \
    }                                                                                                   \
}

#define LAB_ACTIVATION_SOFMAX_DECLARE(Name)                                                             \
struct Name : public torch::nn::Name##Impl, public lab::utils::ActivationModule                         \
{                                                                                                       \
    using torch::nn::Name##Impl::Name##Impl;                                                            \
    Name(int64_t dim = 1) : torch::nn::Name##Impl(dim) {}                                               \
    static constexpr const char* name = #Name;                                                          \
    torch::Tensor forward(const torch::Tensor& input) override                                          \
    {                                                                                                   \
        return torch::nn::Name##Impl::forward(input);                                                   \
    }                                                                                                   \
}

#define LAB_LOSS_FN_DECLARE(Name)                                                                       \
struct Name : public torch::nn::Name##Impl, public lab::utils::LossModule                               \
{                                                                                                       \
    using torch::nn::Name##Impl::Name##Impl;                                                            \
    static constexpr const char* name = #Name;                                                          \
    torch::Tensor forward(const torch::Tensor& input, const torch::Tensor& target) override             \
    {                                                                                                   \
        return torch::nn::Name##Impl::forward(input, target);                                           \
    }                                                                                                   \
}

LAB_ACTIVATION_DECLARE(ReLU);
LAB_ACTIVATION_DECLARE(LeakyReLU);
LAB_ACTIVATION_DECLARE(ELU);
LAB_ACTIVATION_DECLARE(SELU);
LAB_ACTIVATION_DECLARE(SiLU);
LAB_ACTIVATION_DECLARE(Sigmoid);
LAB_ACTIVATION_DECLARE(LogSigmoid);
LAB_ACTIVATION_DECLARE(Tanh);
LAB_ACTIVATION_SOFMAX_DECLARE(Softmax);
LAB_ACTIVATION_SOFMAX_DECLARE(LogSoftmax);
LAB_LOSS_FN_DECLARE(MSELoss);
LAB_LOSS_FN_DECLARE(CrossEntropyLoss);
LAB_LOSS_FN_DECLARE(NLLLoss);
LAB_LOSS_FN_DECLARE(BCELoss);
LAB_LOSS_FN_DECLARE(BCEWithLogitsLoss);
LAB_TYPE_DECLARE(kLinear, torch::enumtype);
LAB_TYPE_DECLARE(kConv1D, torch::enumtype);
LAB_TYPE_DECLARE(kConv2D, torch::enumtype);
LAB_TYPE_DECLARE(kConv3D, torch::enumtype);
LAB_TYPE_DECLARE(kConvTranspose1D, torch::enumtype);
LAB_TYPE_DECLARE(kConvTranspose2D, torch::enumtype);
LAB_TYPE_DECLARE(kConvTranspose3D, torch::enumtype);
LAB_TYPE_DECLARE(kSigmoid, torch::enumtype);
LAB_TYPE_DECLARE(kTanh, torch::enumtype);
LAB_TYPE_DECLARE(kReLU, torch::enumtype);
LAB_TYPE_DECLARE(kLeakyReLU, torch::enumtype);

using Activations = types_t<ReLU, LeakyReLU, ELU, SELU, SiLU, Sigmoid, LogSigmoid, Softmax, LogSoftmax, Tanh>;
using Losses = types_t<MSELoss, CrossEntropyLoss, NLLLoss, BCELoss, BCEWithLogitsLoss>;
using Nets = types_t<lab::agents::MLPNetImpl>;
using NonlinearityTypes = types_t<kLinear, kConv1D, kConv2D, kConv3D, kConvTranspose1D, kConvTranspose2D, kConvTranspose3D, kSigmoid, kTanh, kReLU, kLeakyReLU>;

constexpr named_factory_t<std::shared_ptr<lab::utils::ActivationModule>, shared_ptr_maker, Activations> ActivationFactory;
constexpr named_factory_t<std::shared_ptr<lab::utils::LossModule>, shared_ptr_maker, Losses> LossFactory;
constexpr named_factory_t<std::shared_ptr<lab::agents::NetImpl>, shared_ptr_maker, Nets> NetFactory;
constexpr named_factory_t<torch::nn::init::NonlinearityType , object_maker, NonlinearityTypes> NonlinearityFactory;

torch::nn::Sequential create_fc_model(const std::vector<int64_t>& dims, const std::shared_ptr<lab::utils::ActivationModule>& activation);

torch::nn::Sequential create_fc_model(const std::vector<int64_t>& dims, const std::shared_ptr<lab::utils::LossModule>& loss);

std::shared_ptr<lab::utils::ActivationModule> create_act(std::string_view name);

std::shared_ptr<lab::utils::LossModule> create_loss(std::string_view name);

torch::nn::init::NonlinearityType create_nonlinearirty_type(std::string_view name);

std::shared_ptr<lab::agents::NetImpl> create_net(const NetSpec& spec, int64_t in_dim, int64_t out_dim);

std::shared_ptr<torch::nn::ReLUImpl> create_activation_relu(); 

std::shared_ptr<torch::nn::LeakyReLUImpl> create_activation_leakyrelu(); 

std::shared_ptr<torch::nn::ELUImpl> create_activation_elu(); 

std::shared_ptr<torch::nn::SELUImpl> create_activation_selu(); 

std::shared_ptr<torch::nn::SiLUImpl> create_activation_silu(); 

std::shared_ptr<torch::nn::SigmoidImpl> create_activation_sigmoid(); 

std::shared_ptr<torch::nn::LogSigmoidImpl> create_activation_logsigmoid(); 

std::shared_ptr<torch::nn::SoftmaxImpl> create_activation_softmax(); 

std::shared_ptr<torch::nn::LogSoftmaxImpl> create_activation_logsoftmax(); 

std::shared_ptr<torch::nn::TanhImpl> create_activation_tanh(); 

std::shared_ptr<torch::nn::MSELossImpl> create_mse_loss(); 

std::shared_ptr<torch::nn::CrossEntropyLossImpl> create_cross_entropy_loss(); 

std::shared_ptr<torch::nn::NLLLossImpl> create_nl_loss(); 

std::shared_ptr<torch::nn::BCELossImpl> create_bce_loss(); 

std::shared_ptr<torch::nn::BCEWithLogitsLossImpl> create_bce_with_logits_loss(); 

std::shared_ptr<lab::agents::MLPNetImpl> create_mlp_net(const NetSpec& spec, int64_t in_dim, int64_t out_dim);

}

}






