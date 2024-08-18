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

struct LookaheadOptions : public torch::optim::OptimizerCloneableOptions<LookaheadOptions>
{
public:
    TORCH_ARG(double, lr) = 1e-3;
    TORCH_ARG(int64_t, k) = 5;
    TORCH_ARG(int64_t, step_counter) = 0;
    TORCH_ARG(double, alpha) = 0.5;
    TORCH_ARG(torch::Tensor, slow_weights);
public:
    LookaheadOptions(double lr = 1e-3);
    void serialize(torch::serialize::OutputArchive& archive) const override;
    void serialize(torch::serialize::InputArchive& archive) override;
    double get_lr() const override;
    void set_lr(const double lr) override;
    friend bool operator==(const LookaheadOptions& lhs, const LookaheadOptions& rhs);
};

struct LookaheadParamState : public torch::optim::OptimizerCloneableParamState<LookaheadParamState> 
{
    void serialize(torch::serialize::OutputArchive& archive) const;
    void serialize(torch::serialize::InputArchive& archive);
    friend bool operator==(const LookaheadParamState& lhs, const LookaheadParamState& rhs);
};

template<typename T>
class Lookahead : public torch::optim::Optimizer
{
public:
    TORCH_ARG(T, optimizer);
public:
    explicit Lookahead(std::vector<torch::optim::OptimizerParamGroup> param_groups, LookaheadOptions defaults = {}, torch::optim::OptimizerOptions optim_default = {})
        : torch::optim::Optimizer(std::move(param_groups), std::make_unique<LookaheadOptions>(defaults)), 
          optimizer_(param_groups, static_cast<decltype(T::defaults())>(optim_default))
    {
        static_assert(std::is_base_of_v<torch::optim::Optimizer, T>);

        TORCH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());
        TORCH_CHECK(defaults.alpha() >= 0, "Invalid alpha value: ", defaults.alpha());
        TORCH_CHECK(defaults.step_counter() >= 0, "Invalid step_counter value: ", defaults.step_counter());
        TORCH_CHECK(defaults.k() >= 0, "Invalid k value: ", defaults.k());

        torch::NoGradGuard no_grad;

        state_ = optimizer_.state();
        
        torch::Tensor weights = torch::zeros({
            static_cast<int64_t>(param_groups_.size()), 
            static_cast<int64_t>(param_groups_[0].params().size())}, 
            torch::kDouble
        );
        for (int64_t i = 0; i < param_groups_.size(); i++) 
        {
            auto& options = static_cast<LookaheadOptions&>(param_groups_[i].options());
            options.step_counter(0);

            auto& params = param_groups_[i].params();
            for (int64_t j = 0; j < params.size(); j++) 
                weights.index({i, j}) = params[j].clone().detach();
        }
        auto& defaults_opts = static_cast<LookaheadOptions&>(*defaults_);
        defaults_opts.slow_weights(weights);
    }

    explicit Lookahead(std::vector<torch::Tensor> params, LookaheadOptions defaults = {}, torch::optim::OptimizerOptions optim_default = {})
        : Lookahead({torch::optim::OptimizerParamGroup(std::move(params))}, defaults, optim_default) {}

    torch::Tensor step(torch::optim::Optimizer::LossClosure closure = nullptr) override
    {
        torch::NoGradGuard no_grad;
        torch::Tensor loss = {};
        if (closure != nullptr) 
        {
            at::AutoGradMode enable_grad(true);
            loss = closure();
        }
        loss = optimizer_.step();
        auto& defaults_opts = static_cast<LookaheadOptions&>(*defaults_);
        for (int64_t i = 0; i < param_groups_.size(); i++) 
        {
            auto& options = static_cast<LookaheadOptions&>(param_groups_[i].options());
            options.step_counter(options.step_counter() + 1);
            if(options.step_counter() % defaults_opts.k() != 0)
                continue;

            auto& params = param_groups_[i].params();
            for (int64_t j = 0; j < params.size(); j++) 
            {
                auto& param = params[j];
                if (!param.grad().defined())
                    continue;

                auto q = defaults_opts.slow_weights().index({i, j});
                defaults_opts.slow_weights().index({i, j}).add_(param.data() - q, defaults_opts.alpha());
                param.data().copy_(defaults_opts.slow_weights().index({i, j}));
            }
        }
        return loss;
    }

    void save(torch::serialize::OutputArchive& archive) const
    {
        serialize(*this, archive);
    }

    void load(torch::serialize::InputArchive& archive)
    {
        c10::IValue pytorch_version;
        if (archive.try_read("pytorch_version", pytorch_version)) 
        {
            serialize(*this, archive);
        }
        LAB_UNREACHABLE;
    }

    void share_memory()
    {
        optimizer_.share_memory();
    }
private:
    template <typename Self, typename Archive>
    static void serialize(Self& self, Archive& archive) 
    {
        _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(Lookahead);
    }
};


}
}