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

class NoGradGuard
{
public:
    NoGradGuard();
    virtual ~NoGradGuard();
private:
    std::unique_ptr<torch::NoGradGuard> no_grad_guard;
};

struct Module : public torch::nn::Module
{
    using torch::nn::Module::Module;

    torch::Tensor forward(torch::Tensor input)
    {
        return torch::Tensor();
    }
};

struct ReLU : public torch::nn::ReLUImpl, public Module
{ 
    using torch::nn::ReLUImpl::ReLUImpl;
    static constexpr const char* name = "ReLU";
};
struct LeakyReLU : public torch::nn::LeakyReLUImpl, public Module 
{ 
    using torch::nn::LeakyReLUImpl::LeakyReLUImpl;
    static constexpr const char* name = "LeakyReLU"; 
};
struct ELU : public torch::nn::ELUImpl, public Module 
{ 
    using torch::nn::ELUImpl::ELUImpl;
    static constexpr const char* name = "ELU"; 
};
struct SELU : public torch::nn::SELUImpl, public Module 
{ 
    using torch::nn::SELUImpl::SELUImpl;
    static constexpr const char* name = "SELU"; 
};
struct SiLU : public torch::nn::SiLUImpl, public Module 
{ 
    using torch::nn::SiLUImpl::SiLUImpl;
    static constexpr const char* name = "SiLU"; 
};
struct Sigmoid : public torch::nn::SigmoidImpl, public Module 
{ 
    using torch::nn::SigmoidImpl::SigmoidImpl;
    static constexpr const char* name = "Sigmoid"; 
};
struct LogSigmoid : public torch::nn::LogSigmoidImpl, public Module 
{ 
    using torch::nn::LogSigmoidImpl::LogSigmoidImpl;
    static constexpr const char* name = "LogSigmoid"; 
};
struct Softmax : public torch::nn::SoftmaxImpl, public Module 
{ 
    using torch::nn::SoftmaxImpl::SoftmaxImpl;
    Softmax(int64_t dim = 1) : torch::nn::SoftmaxImpl(dim) {}
    static constexpr const char* name = "Softmax"; 
};
struct LogSoftmax : public torch::nn::LogSoftmaxImpl, public Module 
{ 
    using torch::nn::LogSoftmaxImpl::LogSoftmaxImpl;
    LogSoftmax(int64_t dim = 1) : torch::nn::LogSoftmaxImpl(dim) {}
    static constexpr const char* name = "LogSoftmax"; 
};
struct Tanh : public torch::nn::TanhImpl, public Module 
{ 
    using torch::nn::TanhImpl::TanhImpl;
    static constexpr const char* name = "Tanh"; 
};
struct MSELoss : public torch::nn::MSELossImpl, public Module 
{ 
    using torch::nn::MSELossImpl::MSELossImpl;
    static constexpr const char* name = "MSELoss"; 
};
struct CrossEntropyLoss : public torch::nn::CrossEntropyLossImpl, public Module 
{ 
    using torch::nn::CrossEntropyLossImpl::CrossEntropyLossImpl;
    static constexpr const char* name = "CrossEntropyLoss"; 
};
struct NLLLoss : public torch::nn::NLLLossImpl, public Module 
{ 
    using torch::nn::NLLLossImpl::NLLLossImpl;
    static constexpr const char* name = "NLLLoss"; 
};
struct BCELoss : public torch::nn::BCELossImpl, public Module 
{ 
    using torch::nn::BCELossImpl::BCELossImpl;
    static constexpr const char* name = "BCELoss"; 
};
struct BCEWithLogitsLoss : public torch::nn::BCEWithLogitsLossImpl, public Module 
{ 
    using torch::nn::BCEWithLogitsLossImpl::BCEWithLogitsLossImpl;
    static constexpr const char* name = "BCEWithLogitsLoss"; 
}; 
struct Adam : public torch::optim::Adam 
{ 
    using torch::optim::Adam::Adam;
    static constexpr const char* name = "Adam"; 
};
struct GlobalAdam : public torch::optim::GlobalAdam 
{ 
    using torch::optim::GlobalAdam::GlobalAdam;
    static constexpr const char* name = "GlobalAdam"; 
};
struct RAdam : public torch::optim::RAdam 
{ 
    using torch::optim::RAdam::RAdam;
    static constexpr const char* name = "RAdam"; 
};
struct RMSprop : public torch::optim::RMSprop 
{ 
    using torch::optim::RMSprop::RMSprop;
    static constexpr const char* name = "RMSprop"; 
};
struct GlobalRMSprop : public torch::optim::GlobalRMSprop 
{ 
    using torch::optim::GlobalRMSprop::GlobalRMSprop;
    static constexpr const char* name = "GlobalRMSprop"; 
};
struct StepLR : public torch::optim::StepLR 
{ 
    using torch::optim::StepLR::StepLR;
    static constexpr const char* name = "StepLR"; 
};
struct kLinear : public torch::enumtype::kLinear
{
    using torch::enumtype::kLinear::kLinear;
    static constexpr const char* name = "kLinear";
};
struct kConv1D : public torch::enumtype::kConv1D
{
    using torch::enumtype::kConv1D::kConv1D;
    static constexpr const char* name = "kConv1D";
};
struct kConv2D : public torch::enumtype::kConv2D
{
    using torch::enumtype::kConv2D::kConv2D;
    static constexpr const char* name = "kConv2D";
};
struct kConv3D : public torch::enumtype::kConv3D
{
    using torch::enumtype::kConv3D::kConv3D;
    static constexpr const char* name = "kConv3D";
};
struct kConvTranspose1D : public torch::enumtype::kConvTranspose1D
{
    using torch::enumtype::kConvTranspose1D::kConvTranspose1D;
    static constexpr const char* name = "kConvTranspose1D";
};
struct kConvTranspose2D : public torch::enumtype::kConvTranspose2D
{
    using torch::enumtype::kConvTranspose2D::kConvTranspose2D;
    static constexpr const char* name = "kConvTranspose2D";
};
struct kConvTranspose3D : public torch::enumtype::kConvTranspose3D
{
    using torch::enumtype::kConvTranspose3D::kConvTranspose3D;
    static constexpr const char* name = "kConvTranspose3D";
};
struct kSigmoid : public torch::enumtype::kSigmoid
{
    using torch::enumtype::kSigmoid::kSigmoid;
    static constexpr const char* name = "kSigmoid";
};
struct kTanh : public torch::enumtype::kTanh
{
    using torch::enumtype::kTanh::kTanh;
    static constexpr const char* name = "kTanh";
};
struct kReLU : public torch::enumtype::kReLU
{
    using torch::enumtype::kReLU::kReLU;
    static constexpr const char* name = "kReLU";
};
struct kLeakyReLU : public torch::enumtype::kLeakyReLU
{
    using torch::enumtype::kLeakyReLU::kLeakyReLU;
    static constexpr const char* name = "kLeakyReLU";
};

using Activations = types_t<ReLU, LeakyReLU, ELU, /*SELU,*/ SiLU, Sigmoid, LogSigmoid, Softmax, LogSoftmax, Tanh>;
using Losses = types_t<MSELoss, CrossEntropyLoss, NLLLoss, BCELoss, BCEWithLogitsLoss>;
using Optims = types_t<Adam, GlobalAdam, RAdam, RMSprop, GlobalRMSprop>;
using Schedulars = types_t<StepLR>;
using Nets = types_t<lab::agents::MLPNetImpl>;
using NonlinearityTypes = types_t<kLinear, kConv1D, kConv2D, kConv3D, kConvTranspose1D, kConvTranspose2D, kConvTranspose3D, kSigmoid, kTanh, kReLU, kLeakyReLU>;

constexpr named_factory_t<std::shared_ptr<lab::utils::Module>, shared_ptr_maker, lab::utils::Activations> ActivationFactory;
constexpr named_factory_t<std::shared_ptr<lab::utils::Module>, shared_ptr_maker, lab::utils::Losses> LossFactory;
constexpr named_factory_t<std::shared_ptr<torch::optim::Optimizer>, shared_ptr_maker, lab::utils::Optims> OptimizerFactory;
constexpr named_factory_t<std::shared_ptr<torch::optim::LRScheduler>, shared_ptr_maker, lab::utils::Schedulars> LRSchedularFactory;
constexpr named_factory_t<std::shared_ptr<lab::agents::NetImpl>, shared_ptr_maker, lab::utils::Nets> NetFactory;
constexpr named_factory_t<torch::nn::init::NonlinearityType , object_maker, lab::utils::NonlinearityTypes> NonlinearityFactory;

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





