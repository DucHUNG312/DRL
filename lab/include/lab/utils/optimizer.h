#pragma once

#include "lab/common/common.h"
#include "lab/utils/placeholder.h"
#include "lab/utils/typetraits.h"

namespace torch {
namespace serialize {
class OutputArchive;
class InputArchive;
} // namespace serialize
} // namespace torch

namespace torch {
namespace optim {

class GlobalAdam : public Adam {
 public:
  explicit GlobalAdam(std::vector<OptimizerParamGroup> param_groups, AdamOptions defaults = {})
      : Adam(std::move(param_groups), std::move(defaults)) {}
  explicit GlobalAdam(std::vector<torch::Tensor> params, AdamOptions defaults = {})
      : GlobalAdam({OptimizerParamGroup(std::move(params))}, defaults) {}
  void share_memory();
};

class GlobalRMSprop : public RMSprop {
 public:
  explicit GlobalRMSprop(std::vector<OptimizerParamGroup> param_groups, RMSpropOptions defaults = {})
      : RMSprop(std::move(param_groups), std::move(defaults)) {}
  explicit GlobalRMSprop(std::vector<torch::Tensor> params, RMSpropOptions defaults = {})
      : GlobalRMSprop({OptimizerParamGroup(std::move(params))}, defaults) {}
  void share_memory();
};

struct RAdamOptions : public OptimizerCloneableOptions<RAdamOptions> {
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

struct RAdamParamState : public OptimizerCloneableParamState<RAdamParamState> {
  TORCH_ARG(int64_t, step) = 0;
  TORCH_ARG(torch::Tensor, exp_avg);
  TORCH_ARG(torch::Tensor, exp_avg_sq);

 public:
  void serialize(torch::serialize::InputArchive& archive) override;
  void serialize(torch::serialize::OutputArchive& archive) const override;
  friend bool operator==(const RAdamParamState& lhs, const RAdamParamState& rhs);
};

class RAdam : public Optimizer {
 public:
  explicit RAdam(std::vector<OptimizerParamGroup> param_groups, RAdamOptions defaults = {})
      : Optimizer(std::move(param_groups), std::make_unique<RAdamOptions>(defaults)) {
    TORCH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());
    TORCH_CHECK(defaults.eps() >= 0, "Invalid epsilon value: ", defaults.eps());
    auto betas = defaults.betas();
    TORCH_CHECK(
        0 <= std::get<0>(betas) && std::get<0>(betas) < 1.0, "Invalid beta parameter at index 0: ", std::get<0>(betas));
    TORCH_CHECK(
        0 <= std::get<1>(betas) && std::get<1>(betas) < 1.0, "Invalid beta parameter at index 1: ", std::get<1>(betas));
    TORCH_CHECK(defaults.weight_decay() >= 0, "Invalid weight_decay value: ", defaults.weight_decay());
  }

  explicit RAdam(std::vector<torch::Tensor> params, RAdamOptions defaults = {})
      : RAdam({OptimizerParamGroup(std::move(params))}, defaults) {}

  torch::Tensor step(Optimizer::LossClosure closure = nullptr) override;
  void save(torch::serialize::OutputArchive& archive) const override;
  void load(torch::serialize::InputArchive& archive) override;
  void share_memory();

 private:
  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(RAdam);
  }
};

/*
#define _TORCH_OPTION_LOOKAHEAD_WITH_OPTIMIZER(OptimizerName) \
struct OptimizerName##LookaheadOptions : public
OptimizerCloneableOptions<OptimizerName##LookaheadOptions> \
{ \
public: \
    TORCH_ARG(double, lr) = 1e-3; \
    TORCH_ARG(int64_t, k) = 5; \
    TORCH_ARG(int64_t, step_counter) = 0; \
    TORCH_ARG(double, alpha) = 0.5; \
    TORCH_ARG(torch::Tensor, slow_weights); \
public: \
    friend bool operator==(const OptimizerName##LookaheadOptions& lhs, const
OptimizerName##LookaheadOptions& rhs) \
    { \
        return (lhs.lr() == rhs.lr()) && \
            (lhs.k() == rhs.k()) && \
            (lhs.step_counter() == rhs.step_counter()) && \
            (lhs.alpha() == rhs.alpha()) && \
            (torch::eq(lhs.slow_weights(),
rhs.slow_weights()).all().item<bool>()); \
    } \
                                                                                                                                        \
    void serialize(torch::serialize::OutputArchive& archive) const \
    { \
        _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr); \
        _TORCH_OPTIM_SERIALIZE_TORCH_ARG(k); \
        _TORCH_OPTIM_SERIALIZE_TORCH_ARG(step_counter); \
        _TORCH_OPTIM_SERIALIZE_TORCH_ARG(alpha); \
        _TORCH_OPTIM_SERIALIZE_TORCH_ARG(slow_weights); \
    } \
                                                                                                                                        \
    void serialize(torch::serialize::InputArchive& archive) \
    { \
        _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr); \
        _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(uint64_t, k); \
        _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(uint64_t, step_counter); \
        _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, alpha); \
        _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(torch::Tensor, slow_weights); \
    } \
                                                                                                                                        \
    double get_lr() const \
    { \
        return lr(); \
    } \
                                                                                                                                        \
    void set_lr(const double lr) \
    { \
        this->lr(lr); \
    } \
}

_TORCH_OPTION_LOOKAHEAD_WITH_OPTIMIZER(Adam);
_TORCH_OPTION_LOOKAHEAD_WITH_OPTIMIZER(RAdam);
_TORCH_OPTION_LOOKAHEAD_WITH_OPTIMIZER(RMSprop);

#define _TORCH_PARAM_STATE_LOOKAHEAD_WITH_OPTIMIZER(OptimizerName) \
struct OptimizerName##LookaheadParamState : public
OptimizerCloneableParamState<OptimizerName##LookaheadParamState> \
{ \
    friend bool operator==(const OptimizerName##LookaheadParamState& lhs, const
OptimizerName##LookaheadParamState& rhs)
\
    { \
        return true; \
    } \
                                                                                                                                        \
    void serialize(torch::serialize::OutputArchive& archive) const {} \
                                                                                                                                        \
    void serialize(torch::serialize::InputArchive& archive) {} \
}

_TORCH_PARAM_STATE_LOOKAHEAD_WITH_OPTIMIZER(Adam);
_TORCH_PARAM_STATE_LOOKAHEAD_WITH_OPTIMIZER(RAdam);
_TORCH_PARAM_STATE_LOOKAHEAD_WITH_OPTIMIZER(RMSprop);

#define _TORCH_OPTIM_LOOKAHEAD_WITH_OPTIMIZER(OptimizerName) \
class OptimizerName##Lookahead : public Optimizer \
{ \
public: \
    TORCH_ARG(OptimizerName, optimizer); \
public: \
    explicit OptimizerName##Lookahead(std::vector<OptimizerParamGroup>
param_groups, OptimizerName##LookaheadOptions
defaults = {}, OptimizerName##Options optim_default = {})      \
        : Optimizer(std::move(param_groups),
std::make_unique<OptimizerName##LookaheadOptions>(defaults)), \
          optimizer_(param_groups, optim_default) \
    { \
                                                                                                                                                                                    \
        TORCH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ",
defaults.lr()); \
        TORCH_CHECK(defaults.alpha() >= 0, "Invalid alpha value: ",
defaults.alpha()); \
        TORCH_CHECK(defaults.step_counter() >= 0, "Invalid step_counter value:
", defaults.step_counter()); \
        TORCH_CHECK(defaults.k() >= 0, "Invalid k value: ", defaults.k()); \
                                                                                                                                                                                    \
        torch::NoGradGuard no_grad; \
                                                                                                                                                                                    \
        state_ = optimizer_.state(); \
                                                                                                                                                                                    \
        torch::Tensor weights = torch::zeros({ \
            static_cast<int64_t>(param_groups_.size()), \
            static_cast<int64_t>(param_groups_[0].params().size())}, \
            torch::kDouble \
        ); \
        for (int64_t i = 0; i < param_groups_.size(); i++) \
        { \
            auto& options =
static_cast<OptimizerName##LookaheadOptions&>(param_groups_[i].options()); \
            options.step_counter(0); \
                                                                                                                                                                                    \
            auto& params = param_groups_[i].params(); \
            for (int64_t j = 0; j < params.size(); j++) \
                weights.index({i, j}) = params[j].clone().detach(); \
        } \
        auto& defaults_opts =
static_cast<OptimizerName##LookaheadOptions&>(*defaults_); \
        defaults_opts.slow_weights(weights); \
    } \
                                                                                                                                                                                    \
    explicit OptimizerName##Lookahead(std::vector<torch::Tensor> params,
OptimizerName##LookaheadOptions defaults = {},
OptimizerName##Options optim_default = {})                  \
        : OptimizerName##Lookahead({OptimizerParamGroup(std::move(params))},
defaults, optim_default) {} \
                                                                                                                                                                                    \
    torch::Tensor step(Optimizer::LossClosure closure = nullptr) override \
    { \
        torch::NoGradGuard no_grad; \
        torch::Tensor loss = {}; \
        if (closure != nullptr) \
        { \
            at::AutoGradMode enable_grad(true); \
            loss = closure(); \
        } \
        loss = optimizer_.step(); \
        auto& defaults_opts =
static_cast<OptimizerName##LookaheadOptions&>(*defaults_); \
        for (int64_t i = 0; i < param_groups_.size(); i++) \
        { \
            auto& options =
static_cast<OptimizerName##LookaheadOptions&>(param_groups_[i].options()); \
            options.step_counter(options.step_counter() + 1); \
            if(options.step_counter() % defaults_opts.k() != 0) \
                continue; \
                                                                                                                                                                                    \
            auto& params = param_groups_[i].params(); \
            for (int64_t j = 0; j < params.size(); j++) \
            { \
                auto& param = params[j]; \
                if (!param.grad().defined()) \
                    continue; \
                                                                                                                                                                                    \
                auto q = defaults_opts.slow_weights().index({i, j}); \
                defaults_opts.slow_weights().index({i, j}).add_(param.data() -
q, defaults_opts.alpha()); \
                param.data().copy_(defaults_opts.slow_weights().index({i, j}));
\
            } \
        } \
        return loss; \
    } \
                                                                                                                                                                                    \
    void save(torch::serialize::OutputArchive& archive) const \
    { \
        serialize(*this, archive); \
    } \
                                                                                                                                                                                    \
    void load(torch::serialize::InputArchive& archive) \
    { \
        c10::IValue pytorch_version; \
        if (archive.try_read("pytorch_version", pytorch_version)) \
        { \
            serialize(*this, archive); \
        } \
        LAB_UNREACHABLE; \
    } \
                                                                                                                                                                                    \
private: \
    template <typename Self, typename Archive> \
    static void serialize(Self& self, Archive& archive) \
    { \
        _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(OptimizerName##Lookahead); \
    } \
}

_TORCH_OPTIM_LOOKAHEAD_WITH_OPTIMIZER(Adam);
_TORCH_OPTIM_LOOKAHEAD_WITH_OPTIMIZER(RAdam);
_TORCH_OPTIM_LOOKAHEAD_WITH_OPTIMIZER(RMSprop);
*/

struct AnyOptimPlaceholder : public lab::utils::Placeholder {
  using Placeholder::Placeholder;

  virtual std::shared_ptr<Optimizer> ptr() = 0;

  virtual std::unique_ptr<AnyOptimPlaceholder> copy() const = 0;
};

template <typename OptimType>
struct AnyOptimHolder : public AnyOptimPlaceholder {
  explicit AnyOptimHolder(std::shared_ptr<OptimType>&& optim_)
      : AnyOptimPlaceholder(typeid(OptimType)), optim(std::move(optim_)) {}

  std::shared_ptr<Optimizer> ptr() override {
    return optim;
  }

  std::unique_ptr<AnyOptimPlaceholder> copy() const override {
    return std::make_unique<AnyOptimHolder>(*this);
  }

  std::shared_ptr<OptimType> optim;
};

class AnyOptim {
 public:
  AnyOptim() = default;

  template <typename OptimType>
  explicit AnyOptim(std::shared_ptr<OptimType> Optim);

  template <typename OptimType, typename = lab::utils::enable_if_optim_t<OptimType>>
  explicit AnyOptim(OptimType&& Optim);

  AnyOptim(AnyOptim&&) = default;
  AnyOptim& operator=(AnyOptim&&) = default;

  AnyOptim(const AnyOptim& other);
  AnyOptim& operator=(const AnyOptim& other);

  template <typename OptimType>
  AnyOptim& operator=(std::shared_ptr<OptimType> optim);

  template <typename T, typename = lab::utils::enable_if_optim_t<T>>
  T& get();

  template <typename T, typename = lab::utils::enable_if_optim_t<T>>
  const T& get() const;

  template <typename T, typename ContainedType = typename T::ContainedType>
  T get() const;

  std::shared_ptr<Optimizer> ptr() const;

  template <typename T, typename = lab::utils::enable_if_optim_t<T>>
  std::shared_ptr<T> ptr() const;

  const std::type_info& type_info() const;

  bool is_empty() const noexcept;

 private:
  template <typename OptimType>
  std::unique_ptr<AnyOptimPlaceholder> make_holder(std::shared_ptr<OptimType>&& optim);

  template <typename OptimType>
  OptimType& get_() const;

  std::unique_ptr<AnyOptimPlaceholder> content_;
};

template <typename OptimType>
AnyOptim::AnyOptim(std::shared_ptr<OptimType> optim) : content_(make_holder(std::move(optim))) {
  static_assert(lab::utils::is_optim<OptimType>::value);
  static_assert(lab::utils::has_step<OptimType>::value);
}

template <typename OptimType, typename>
AnyOptim::AnyOptim(OptimType&& optim) : AnyOptim(std::make_shared<OptimType>(std::forward<OptimType>(optim))) {}

inline AnyOptim::AnyOptim(const AnyOptim& other) : content_(other.content_ ? other.content_->copy() : nullptr) {}

inline AnyOptim& AnyOptim::operator=(const AnyOptim& other) {
  if (this != &other)
    content_ = other.content_ ? other.content_->copy() : nullptr;
  return *this;
}

template <typename OptimType>
AnyOptim& AnyOptim::operator=(std::shared_ptr<OptimType> optim) {
  return (*this = AnyOptim(std::move(optim)));
}

template <typename T, typename>
T& AnyOptim::get() {
  TORCH_CHECK(!is_empty(), "Cannot call get() on an empty AnyOptim");
  return get_<T>();
}

template <typename T, typename>
const T& AnyOptim::get() const {
  TORCH_CHECK(!is_empty(), "Cannot call get() on an empty AnyOptim");
  return get_<T>();
}

template <typename T, typename ContainedType>
T AnyOptim::get() const {
  return T(ptr<ContainedType>());
}

inline std::shared_ptr<Optimizer> AnyOptim::ptr() const {
  TORCH_CHECK(!is_empty(), "Cannot call ptr() on an empty AnyOptim");
  return content_->ptr();
}

template <typename T, typename>
std::shared_ptr<T> AnyOptim::ptr() const {
  TORCH_CHECK(!is_empty(), "Cannot call ptr() on an empty AnyOptim");
  get_<T>();
  return std::dynamic_pointer_cast<T>(ptr());
}

inline const std::type_info& AnyOptim::type_info() const {
  TORCH_CHECK(!is_empty(), "Cannot call type_info() on an empty AnyOptim");
  return content_->type_info;
}

inline bool AnyOptim::is_empty() const noexcept {
  return content_ == nullptr;
}

template <typename OptimType>
std::unique_ptr<AnyOptimPlaceholder> AnyOptim::make_holder(std::shared_ptr<OptimType>&& optim) {
  return std::make_unique<AnyOptimHolder<std::decay_t<OptimType>>>(std::move(optim));
}

template <typename OptimType>
OptimType& AnyOptim::get_() const {
  using M = typename std::remove_reference<OptimType>::type;
  static_assert(lab::utils::has_step<M>::value, "Can only call AnyOptim::get<T> with a type T that has a step method");
  if (typeid(OptimType).hash_code() == type_info().hash_code()) {
    return *static_cast<AnyOptimHolder<OptimType>&>(*content_).optim;
  }
  AT_ERROR(
      "Attempted to cast optimizer of type ",
      c10::demangle(type_info().name()),
      " to type ",
      c10::demangle(typeid(OptimType).name()));
}

} // namespace optim
} // namespace torch

namespace lab {

namespace utils {

LAB_TYPE_DECLARE(Adam, torch::optim);
LAB_TYPE_DECLARE(GlobalAdam, torch::optim);
LAB_TYPE_DECLARE(RAdam, torch::optim);
LAB_TYPE_DECLARE(RMSprop, torch::optim);
LAB_TYPE_DECLARE(GlobalRMSprop, torch::optim);
LAB_TYPE_DECLARE(StepLR, torch::optim);

using Optims = types_t<Adam, GlobalAdam, RAdam, RMSprop, GlobalRMSprop>;
using Schedulars = types_t<StepLR>;

constexpr named_factory_t<std::shared_ptr<torch::optim::Optimizer>, shared_ptr_maker, Optims> OptimizerFactory;
constexpr named_factory_t<std::shared_ptr<torch::optim::LRScheduler>, shared_ptr_maker, Schedulars> LRSchedularFactory;

std::shared_ptr<torch::optim::Optimizer> create_optim(std::string_view name, const std::vector<torch::Tensor>& params);

std::shared_ptr<torch::optim::LRScheduler> create_lr_schedular(
    const std::shared_ptr<torch::optim::Optimizer>& optimizer,
    const LrSchedulerSpec& spec);

std::shared_ptr<torch::optim::Adam> create_optim_adam(const std::vector<torch::Tensor>& params);

std::shared_ptr<torch::optim::GlobalAdam> create_optim_global_adam(const std::vector<torch::Tensor>& params);

std::shared_ptr<torch::optim::RAdam> create_optim_radam(const std::vector<torch::Tensor>& params);

std::shared_ptr<torch::optim::RMSprop> create_optim_rmsprop(const std::vector<torch::Tensor>& params);

std::shared_ptr<torch::optim::GlobalRMSprop> create_optim_global_rmsprop(const std::vector<torch::Tensor>& params);

std::shared_ptr<torch::optim::StepLR> create_lr_schedular_step(
    const std::shared_ptr<torch::optim::Optimizer>& optimizer,
    const LrSchedulerSpec& spec);

} // namespace utils

} // namespace lab