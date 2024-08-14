#pragma once

#include "lab/core.h"

#include <torch/nn/module.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/serialize.h>

namespace torch {
namespace serialize {
class OutputArchive;
class InputArchive;
} // namespace serialize
} // namespace torch

namespace lab
{
namespace utils
{

class GlobalAdam : public torch::optim::Adam
{
public:
    explicit GlobalAdam(std::vector<torch::optim::OptimizerParamGroup> param_groups, torch::optim::AdamOptions defaults /*= {}*/)
        : torch::optim::Adam(std::move(param_groups), std::move(defaults)) {}
    explicit GlobalAdam(std::vector<torch::Tensor> params, torch::optim::AdamOptions defaults /*= {}*/)
        : GlobalAdam({torch::optim::OptimizerParamGroup(std::move(params))}, defaults) {}
    void share_memory();
};

class GlobalRMSprop : public torch::optim::RMSprop
{
public:
    explicit GlobalRMSprop(std::vector<torch::optim::OptimizerParamGroup> param_groups, torch::optim::RMSpropOptions defaults /*= {}*/)
        : torch::optim::RMSprop(std::move(param_groups), std::move(defaults)) {}
    explicit GlobalRMSprop(std::vector<torch::Tensor> params, torch::optim::RMSpropOptions defaults /*= {}*/)
        : GlobalRMSprop({torch::optim::OptimizerParamGroup(std::move(params))}, defaults) {}
    void share_memory();
};

struct RAdamOptions : public torch::optim::OptimizerCloneableOptions<RAdamOptions> 
{
    RAdamOptions(double lr = 1e-3);

    TORCH_ARG(double, lr) = 1e-3;
    typedef std::tuple<double, double> betas_t;
    TORCH_ARG(betas_t, betas) = std::make_tuple(0.9, 0.999);
    TORCH_ARG(double, eps) = 1e-8;
    TORCH_ARG(double, weight_decay) = 0;
    TORCH_ARG(torch::Tensor, buffer) = torch::zeros({10, 3});
public:
    void serialize(torch::serialize::InputArchive& archive) override;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    friend bool operator==(const RAdamOptions& lhs, const RAdamOptions& rhs);
    double get_lr() const override;
    void set_lr(const double lr) override;
};

struct RAdamParamState : public torch::optim::OptimizerCloneableParamState<RAdamParamState> 
{
    TORCH_ARG(int64_t, step) = 0;
    TORCH_ARG(torch::Tensor, exp_avg);
    TORCH_ARG(torch::Tensor, exp_avg_sq);
public:
    void serialize(torch::serialize::InputArchive& archive) override;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    friend bool operator==(const RAdamParamState& lhs, const RAdamParamState& rhs);
};

class RAdam : public torch::optim::Optimizer
{
public:
    explicit RAdam(std::vector<torch::optim::OptimizerParamGroup> param_groups, RAdamOptions defaults = {})
        : torch::optim::Optimizer(std::move(param_groups), std::make_unique<RAdamOptions>(defaults)) 
    {
        TORCH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());
        TORCH_CHECK(defaults.eps() >= 0, "Invalid epsilon value: ", defaults.eps());
        auto betas = defaults.betas();
        TORCH_CHECK(0 <= std::get<0>(betas) && std::get<0>(betas) < 1.0, "Invalid beta parameter at index 0: ", std::get<0>(betas));
        TORCH_CHECK(0 <= std::get<1>(betas) && std::get<1>(betas) < 1.0, "Invalid beta parameter at index 1: ", std::get<1>(betas));
        TORCH_CHECK(defaults.weight_decay() >= 0, "Invalid weight_decay value: ", defaults.weight_decay());
    }

    explicit RAdam(std::vector<torch::Tensor> params, RAdamOptions defaults = {})
        : RAdam({torch::optim::OptimizerParamGroup(std::move(params))}, defaults) {}

    torch::Tensor step(torch::optim::Optimizer::LossClosure closure = nullptr) override;
    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;
    void share_memory();
private:
    template <typename Self, typename Archive>
    static void serialize(Self& self, Archive& archive) 
    {
        _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(RAdam);
    }
};


}
}