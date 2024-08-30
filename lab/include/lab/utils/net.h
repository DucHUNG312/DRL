#pragma once

#include "lab/core.h"
#include "lab/utils/optimizer.h"
#include "lab/utils/spec.h"
#include "lab/utils/typetraits.h"

namespace lab
{
namespace agents
{
class NetImpl;
class MLPNetImpl;
}

namespace utils
{

struct Module : public torch::nn::Module
{
    using torch::nn::Module::Module;
    torch::Tensor forward(torch::Tensor input);
};

class NoGradGuard
{
public:
    NoGradGuard();
    virtual ~NoGradGuard();
private:
    std::unique_ptr<torch::NoGradGuard> no_grad_guard;
};

#define LAB_ACTIVATION_DECLARE(Name)                                      \
struct Name : public torch::nn::Name##Impl, public lab::utils::Module     \
{                                                                         \
    using torch::nn::Name##Impl::Name##Impl;                              \
    static constexpr const char* name = #Name;                            \
}

#define LAB_ACTIVATION_SOFMAX_DECLARE(Name)                               \
struct Name : public torch::nn::Name##Impl, public lab::utils::Module     \
{                                                                         \
    using torch::nn::Name##Impl::Name##Impl;                              \
    Name(int64_t dim = 1) : torch::nn::Name##Impl(dim) {}                 \
    static constexpr const char* name = #Name;                            \
}

LAB_ACTIVATION_DECLARE(ReLU);
LAB_ACTIVATION_DECLARE(LeakyReLU);
LAB_ACTIVATION_DECLARE(ELU);
LAB_ACTIVATION_DECLARE(SELU);
LAB_ACTIVATION_DECLARE(SiLU);
LAB_ACTIVATION_DECLARE(Sigmoid);
LAB_ACTIVATION_DECLARE(LogSigmoid);
LAB_ACTIVATION_DECLARE(Tanh);
LAB_ACTIVATION_DECLARE(MSELoss);
LAB_ACTIVATION_DECLARE(CrossEntropyLoss);
LAB_ACTIVATION_DECLARE(NLLLoss);
LAB_ACTIVATION_DECLARE(BCELoss);
LAB_ACTIVATION_DECLARE(BCEWithLogitsLoss);
LAB_ACTIVATION_SOFMAX_DECLARE(Softmax);
LAB_ACTIVATION_SOFMAX_DECLARE(LogSoftmax);
LAB_TYPE_DECLARE(Adam, torch::optim);
LAB_TYPE_DECLARE(GlobalAdam, torch::optim);
LAB_TYPE_DECLARE(RAdam, torch::optim);
LAB_TYPE_DECLARE(RMSprop, torch::optim);
LAB_TYPE_DECLARE(GlobalRMSprop, torch::optim);
LAB_TYPE_DECLARE(StepLR, torch::optim);
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

using Activations = types_t<ReLU, LeakyReLU, ELU, /*SELU,*/ SiLU, Sigmoid, LogSigmoid, Softmax, LogSoftmax, Tanh>;
using Losses = types_t<MSELoss, CrossEntropyLoss, NLLLoss, BCELoss, BCEWithLogitsLoss>;
using Optims = types_t<Adam, GlobalAdam, RAdam, RMSprop, GlobalRMSprop>;
using Schedulars = types_t<StepLR>;
using Nets = types_t<lab::agents::MLPNetImpl>;
using NonlinearityTypes = types_t<kLinear, kConv1D, kConv2D, kConv3D, kConvTranspose1D, kConvTranspose2D, kConvTranspose3D, kSigmoid, kTanh, kReLU, kLeakyReLU>;

constexpr named_factory_t<std::shared_ptr<lab::utils::Module>, shared_ptr_maker, Activations> ActivationFactory;
constexpr named_factory_t<std::shared_ptr<lab::utils::Module>, shared_ptr_maker, Losses> LossFactory;
constexpr named_factory_t<std::shared_ptr<torch::optim::Optimizer>, shared_ptr_maker, Optims> OptimizerFactory;
constexpr named_factory_t<std::shared_ptr<torch::optim::LRScheduler>, shared_ptr_maker, Schedulars> LRSchedularFactory;
constexpr named_factory_t<std::shared_ptr<lab::agents::NetImpl>, shared_ptr_maker, Nets> NetFactory;
constexpr named_factory_t<torch::nn::init::NonlinearityType , object_maker, NonlinearityTypes> NonlinearityFactory;

torch::nn::Sequential create_fc_model(const std::vector<int64_t>& dims, const std::shared_ptr<lab::utils::Module>& activation);

std::shared_ptr<lab::utils::Module> create_act(std::string_view name);

std::shared_ptr<lab::utils::Module> create_loss(std::string_view name);

std::shared_ptr<torch::optim::Optimizer> create_optim(std::string_view name, const std::vector<torch::Tensor>& params);

std::shared_ptr<torch::optim::LRScheduler> create_lr_schedular(const std::shared_ptr<torch::optim::Optimizer>& optimizer, const LrSchedulerSpec& spec);

torch::nn::init::NonlinearityType create_nonlinearirty_type(std::string_view name);

std::shared_ptr<lab::agents::NetImpl> create_net(const utils::NetSpec& spec, int64_t in_dim, int64_t out_dim);

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

std::shared_ptr<torch::optim::Adam> create_optim_adam(const std::vector<torch::Tensor>& params); 

std::shared_ptr<torch::optim::GlobalAdam> create_optim_global_adam(const std::vector<torch::Tensor>& params);

std::shared_ptr<torch::optim::RAdam> create_optim_radam(const std::vector<torch::Tensor>& params);

std::shared_ptr<torch::optim::RMSprop> create_optim_rmsprop(const std::vector<torch::Tensor>& params);

std::shared_ptr<torch::optim::GlobalRMSprop> create_optim_global_rmsprop(const std::vector<torch::Tensor>& params);

std::shared_ptr<torch::optim::StepLR> create_lr_schedular_step(const std::shared_ptr<torch::optim::Optimizer>& optimizer, const LrSchedulerSpec& spec);

std::shared_ptr<lab::agents::MLPNetImpl> create_mlp_net(const utils::NetSpec& spec, int64_t in_dim, int64_t out_dim);

}

}






