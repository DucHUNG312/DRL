#include "lab/utils/optimizer.h"
#include "lab/utils/math.h"

namespace lab
{
namespace utils
{

void GlobalAdam::share_memory()
{
    for (torch::optim::OptimizerParamGroup& group : param_groups_)
    {
        auto& options = static_cast<torch::optim::AdamOptions&>(group.options());
        for(torch::Tensor& param : group.params())
        {
            if (state_.find(param.unsafeGetTensorImpl()) != state_.end())
            {
                auto& state = static_cast<torch::optim::AdamParamState&>(*state_[param.unsafeGetTensorImpl()]);
                state.exp_avg().share_memory_();
                state.exp_avg_sq().share_memory_();
                if (options.amsgrad()) 
                    state.max_exp_avg_sq().share_memory_();
            }
        }
    }
        
}

void GlobalRMSprop::share_memory()
{
    for (torch::optim::OptimizerParamGroup& group : param_groups_)
    {
        auto& options = static_cast<torch::optim::RMSpropOptions&>(group.options());
        for(torch::Tensor& param : group.params())
        {
            if (state_.find(param.unsafeGetTensorImpl()) != state_.end())
            {
                auto& state = static_cast<torch::optim::RMSpropParamState&>(*state_.at(param.unsafeGetTensorImpl()));
                state.square_avg().share_memory_();
                if (options.momentum() > 0)
                    state.momentum_buffer().share_memory_();
                if (options.centered())
                    state.grad_avg().share_memory_();
            }
        }
    }
}

RAdamOptions::RAdamOptions(double lr) : lr_(lr) {}

bool operator==(const RAdamOptions& lhs, const RAdamOptions& rhs) 
{
    return (lhs.lr() == rhs.lr()) &&
        (std::get<0>(lhs.betas()) == std::get<0>(rhs.betas())) &&
        (std::get<1>(lhs.betas()) == std::get<1>(rhs.betas())) &&
        (lhs.eps() == rhs.eps()) &&
        (lhs.weight_decay() == rhs.weight_decay()) &&
        (torch::all(lhs.buffer() == rhs.buffer()).item<bool>());
}

void RAdamOptions::serialize(torch::serialize::OutputArchive& archive) const 
{
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(betas);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(eps);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(weight_decay);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(buffer);
}

void RAdamOptions::serialize(torch::serialize::InputArchive& archive) 
{
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(betas_t, betas);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, eps);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, weight_decay);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(torch::Tensor, buffer);
}

double RAdamOptions::get_lr() const 
{
    return lr();
}

void RAdamOptions::set_lr(const double lr) 
{
    this->lr(lr);
}

bool operator==(const RAdamParamState& lhs, const RAdamParamState& rhs) 
{
    return (lhs.step() == rhs.step()) &&
        torch::equal(lhs.exp_avg(), rhs.exp_avg()) &&
        torch::equal(lhs.exp_avg_sq(), rhs.exp_avg_sq());
}

void RAdamParamState::serialize(torch::serialize::OutputArchive& archive) const 
{
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(step);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_avg);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_avg_sq);
}

void RAdamParamState::serialize(torch::serialize::InputArchive& archive) 
{
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, step);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(torch::Tensor, exp_avg);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(torch::Tensor, exp_avg_sq);
}

torch::Tensor RAdam::step(torch::optim::Optimizer::LossClosure closure /*= nullptr*/)
{
    torch::NoGradGuard no_grad;
    torch::Tensor loss = {};
    if (closure != nullptr) 
    {
        at::AutoGradMode enable_grad(true);
        loss = closure();
    }
    for (auto& group : param_groups_) 
    {
        for (auto& p : group.params()) 
        {
            if (!p.grad().defined())
                continue;

            auto grad = p.grad();

            TORCH_CHECK(!grad.is_sparse(), "RAdam does not support sparse gradients" /*, please consider SparseAdam instead*/);
            
            auto p_data_fp32 = p.to(torch::kFloat);
            auto param_state = state_.find(p.unsafeGetTensorImpl());
            auto& options = static_cast<RAdamOptions&>(group.options());

            // State initialization
            if (param_state == state_.end()) 
            {
                auto state = std::make_unique<RAdamParamState>();
                state->step(0);
                state->exp_avg(torch::zeros_like(p, torch::MemoryFormat::Preserve));
                state->exp_avg_sq(torch::zeros_like(p, torch::MemoryFormat::Preserve));
                state_[p.unsafeGetTensorImpl()] = std::move(state);
            }

            auto& state = static_cast<RAdamParamState&>(*state_[p.unsafeGetTensorImpl()]);
            auto exp_avg = state.exp_avg().to(p_data_fp32.dtype());
            auto exp_avg_sq = state.exp_avg_sq().to(p_data_fp32.dtype());
            auto beta1 = std::get<0>(options.betas());
            auto beta2 = std::get<1>(options.betas());

            // Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, 1 - beta1);
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, 1 - beta2);

            state.step(state.step() + 1);
            torch::Tensor buffered = options.buffer()[int64_t(state.step() % 10)];
            double N_sma;
            double step_size; 
            if(state.step() == buffered[0].item<int64_t>())
            {
                N_sma = buffered[1].item<double>();
                step_size = buffered[2].item<double>();
            }
            else
            {
                buffered[0].fill_(state.step());
                double bias_correction1 = 1 - std::pow(beta1, state.step());
                double bias_correction2 = 1 - std::pow(beta2, state.step());
                double N_sma_max = 2 / (1 - beta2) - 1;
                N_sma = N_sma_max - 2 * state.step() * bias_correction2 / (1 - bias_correction2);
                buffered[1].fill_(N_sma);
                if(N_sma >= 5)
                    step_size = utils::math::safe_sqrt((1 - bias_correction2) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / bias_correction1;
                else
                    step_size = 1 / bias_correction1;
                buffered[2].fill_(step_size);
            }

            if (options.weight_decay() != 0)
                p_data_fp32.add_(p_data_fp32, -options.weight_decay() * options.lr());
            
            auto adap_lr = (-step_size * options.lr());

            if(N_sma >= 5)
            {
                auto denom = exp_avg_sq.sqrt().add_(options.eps());
                p_data_fp32.addcdiv_(exp_avg, denom, adap_lr);
            }
            else
                p_data_fp32.add_(exp_avg, adap_lr);
            
            p.copy_(p_data_fp32);
        }
    }
    return loss;
}

void RAdam::save(torch::serialize::OutputArchive& archive) const
{
    serialize(*this, archive);
}

void RAdam::load(torch::serialize::InputArchive& archive)
{
    c10::IValue pytorch_version;
    if (archive.try_read("pytorch_version", pytorch_version)) 
    {
        serialize(*this, archive);
    }
    LAB_UNREACHABLE;
}

void RAdam::share_memory()
{
    for (torch::optim::OptimizerParamGroup& group : param_groups_)
    {
        auto& options = static_cast<RAdamOptions&>(group.options());
        for(torch::Tensor& param : group.params())
        {
            if (state_.find(param.unsafeGetTensorImpl()) != state_.end())
            {
                auto& state = static_cast<RAdamParamState&>(*state_.at(param.unsafeGetTensorImpl()));
                state.exp_avg().share_memory_();
                state.exp_avg_sq().share_memory_();
            }
        }
    }
}

}
}