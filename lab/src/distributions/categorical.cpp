#include "lab/distributions/categorical.h"
#include "lab/distributions/utils.h"

namespace lab {
namespace distributions {

Categorical::Categorical(const torch::Tensor& in, bool is_logits /*= false*/)
    : Distribution(in.sizes()), is_logits_(is_logits) {
  std::vector<int64_t> shape_vec = extended_shape();
  params_ = is_logits_ ? (in - in.logsumexp({-1}, true)) : (in / in.sum({-1}, true));
  num_events_ = params_.size(-1);
  mean_ = torch::full(
      torch::IntArrayRef(shape_vec),
      std::numeric_limits<double>::quiet_NaN(),
      torch::TensorOptions().dtype(torch::kDouble).device(params_.device()));
  variance_ = torch::full(
      torch::IntArrayRef(shape_vec),
      std::numeric_limits<double>::quiet_NaN(),
      torch::TensorOptions().dtype(torch::kDouble).device(params_.device()));
}

torch::Tensor Categorical::logits() {
  return is_logits_ ? params_ : probs_to_logits(params_);
}

torch::Tensor Categorical::probs() {
  return is_logits_ ? logits_to_probs(params_) : params_;
}

torch::IntArrayRef Categorical::params_shape() {
  return params_.sizes();
}

torch::Tensor Categorical::sample(torch::IntArrayRef sample_shape /*= {}*/) {
  int64_t num_samples = 1;
  for (int64_t dim : sample_shape)
    num_samples *= dim;

  torch::Tensor probs_2d = probs().view({-1, num_events_});
  torch::Tensor samples_2d = torch::multinomial(probs_2d, num_samples, true).transpose(0, 1);
  return samples_2d;
}

torch::Tensor Categorical::log_prob(const torch::Tensor& value) {
  auto broadcasted_tensors = torch::broadcast_tensors({value.to(torch::kInt64).unsqueeze(-1), logits()});
  torch::Tensor val = broadcasted_tensors[0].index({torch::indexing::Ellipsis, torch::indexing::Slice(0, 1)});
  torch::Tensor log_pmf = broadcasted_tensors[1];
  return log_pmf.gather(-1, val).squeeze(-1);
}

torch::Tensor Categorical::entropy() {
  double min_real = std::numeric_limits<double>::min();
  torch::Tensor clamped_logits = torch::clamp(logits(), min_real);
  torch::Tensor p_log_p = clamped_logits * probs();
  return -p_log_p.sum(-1);
}

torch::Tensor Categorical::rsample(torch::IntArrayRef sample_shape /*= {}*/) {
  LAB_UNIMPLEMENTED;
  return torch::Tensor();
}

torch::Tensor Categorical::sample_n(int64_t n) {
  LAB_UNIMPLEMENTED;
  return torch::Tensor();
}

torch::Tensor Categorical::cdf(const torch::Tensor& value) {
  LAB_UNIMPLEMENTED;
  return torch::Tensor();
}

torch::Tensor Categorical::icdf(const torch::Tensor& value) {
  LAB_UNIMPLEMENTED;
  return torch::Tensor();
}

} // namespace distributions
} // namespace lab