#include "lab/utils/net.h"
#include "lab/agents/net/mlp.h"
#include "lab/utils/spec.h"

namespace lab {

namespace utils {

NoGradGuard::NoGradGuard() {
  no_grad_guard = std::make_unique<torch::NoGradGuard>();
}

NoGradGuard::~NoGradGuard() {
  no_grad_guard.reset();
}

namespace internal {
template <typename T>
torch::nn::Sequential create_fc_model(const std::vector<int64_t>& dims, const std::shared_ptr<T>& activation) {
  LAB_CHECK(dims.size() >= 2, "Expected dim greater or equal to 2, got " << dims.size());
  torch::nn::Sequential model;
  for (size_t i = 0; i < dims.size() - 1; i++) {
    auto linear = torch::nn::Linear(dims[i], dims[i + 1]);
    model->push_back(linear);
    // auto act = activation->clone();
    model->push_back(activation);
  }
  return model;
}
} // namespace internal

torch::nn::Sequential create_fc_model(
    const std::vector<int64_t>& dims,
    const std::shared_ptr<lab::utils::ActivationModule>& activation) {
  return internal::create_fc_model(dims, activation);
}

torch::nn::Sequential create_fc_model(
    const std::vector<int64_t>& dims,
    const std::shared_ptr<lab::utils::LossModule>& loss) {
  return internal::create_fc_model(dims, loss);
}

std::shared_ptr<lab::utils::ActivationModule> create_act(std::string_view name) {
  return ActivationFactory(name);
}

std::shared_ptr<lab::utils::LossModule> create_loss(std::string_view name) {
  return LossFactory(name);
}

torch::nn::init::NonlinearityType create_nonlinearirty_type(std::string_view name) {
  return NonlinearityFactory(name);
}

std::shared_ptr<lab::agents::NetImpl> create_net(const NetSpec& spec, int64_t in_dim, int64_t out_dim) {
  return NetFactory(spec.name, std::move(spec), in_dim, out_dim);
}

std::shared_ptr<torch::nn::ReLUImpl> create_activation_relu() {
  return std::dynamic_pointer_cast<torch::nn::ReLUImpl>(create_act("ReLU"));
}

std::shared_ptr<torch::nn::LeakyReLUImpl> create_activation_leakyrelu() {
  return std::dynamic_pointer_cast<torch::nn::LeakyReLUImpl>(create_act("LeakyReLU"));
}

std::shared_ptr<torch::nn::ELUImpl> create_activation_elu() {
  return std::dynamic_pointer_cast<torch::nn::ELUImpl>(create_act("ELU"));
}

std::shared_ptr<torch::nn::SELUImpl> create_activation_selu() {
  return std::dynamic_pointer_cast<torch::nn::SELUImpl>(create_act("SELU"));
}

std::shared_ptr<torch::nn::SiLUImpl> create_activation_silu() {
  return std::dynamic_pointer_cast<torch::nn::SiLUImpl>(create_act("SiLU"));
}

std::shared_ptr<torch::nn::SigmoidImpl> create_activation_sigmoid() {
  return std::dynamic_pointer_cast<torch::nn::SigmoidImpl>(create_act("Sigmoid"));
}

std::shared_ptr<torch::nn::LogSigmoidImpl> create_activation_logsigmoid() {
  return std::dynamic_pointer_cast<torch::nn::LogSigmoidImpl>(create_act("LogSigmoid"));
}

std::shared_ptr<torch::nn::SoftmaxImpl> create_activation_softmax() {
  return std::dynamic_pointer_cast<torch::nn::SoftmaxImpl>(create_act("Softmax"));
}

std::shared_ptr<torch::nn::LogSoftmaxImpl> create_activation_logsoftmax() {
  return std::dynamic_pointer_cast<torch::nn::LogSoftmaxImpl>(create_act("LogSoftmax"));
}

std::shared_ptr<torch::nn::TanhImpl> create_activation_tanh() {
  return std::dynamic_pointer_cast<torch::nn::TanhImpl>(create_act("Tanh"));
}

std::shared_ptr<torch::nn::MSELossImpl> create_mse_loss() {
  return std::dynamic_pointer_cast<torch::nn::MSELossImpl>(create_loss("MSELoss"));
}

std::shared_ptr<torch::nn::CrossEntropyLossImpl> create_cross_entropy_loss() {
  return std::dynamic_pointer_cast<torch::nn::CrossEntropyLossImpl>(create_loss("CrossEntropyLoss"));
}

std::shared_ptr<torch::nn::NLLLossImpl> create_nl_loss() {
  return std::dynamic_pointer_cast<torch::nn::NLLLossImpl>(create_loss("NLLLoss"));
}

std::shared_ptr<torch::nn::BCELossImpl> create_bce_loss() {
  return std::dynamic_pointer_cast<torch::nn::BCELossImpl>(create_loss("BCELoss"));
}

std::shared_ptr<torch::nn::BCEWithLogitsLossImpl> create_bce_with_logits_loss() {
  return std::dynamic_pointer_cast<torch::nn::BCEWithLogitsLossImpl>(create_loss("BCEWithLogitsLoss"));
}

std::shared_ptr<lab::agents::MLPNetImpl> create_mlp_net(const NetSpec& spec, int64_t in_dim, int64_t out_dim) {
  return std::dynamic_pointer_cast<lab::agents::MLPNetImpl>(create_net(spec, in_dim, out_dim));
}

} // namespace utils

} // namespace lab