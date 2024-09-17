#pragma once

#include "lab/spaces/any.h"
#include "lab/spaces/base.h"

namespace lab {
namespace spaces {

struct BoxOptions {
  BoxOptions(const torch::Tensor& low, const torch::Tensor& high);
  LAB_ARG(torch::Tensor, low);
  LAB_ARG(torch::Tensor, high);
};

class BoxImpl : public ClonableSpace<BoxImpl> {
 public:
  BoxImpl() = default;
  explicit BoxImpl(const BoxOptions& options_);
  explicit BoxImpl(const torch::Tensor& low, const torch::Tensor& high) : BoxImpl(BoxOptions(low, high)) {}

  void reset() override;

  torch::Tensor sample(/*const torch::Tensor& mask*/);

  bool contains(const torch::Tensor& x) const;

  // Overload for zero arguments
  bool contains() const;

 public:
  BoxOptions options;
  torch::Tensor low;
  torch::Tensor high;
};

LAB_SPACE(Box);

Box make_box_space(torch::IntArrayRef shape = torch::IntArrayRef(1));

Box make_box_space(const torch::Tensor& low, const torch::Tensor& high);

std::shared_ptr<BoxImpl> make_box_space_impl(const torch::Tensor& low, const torch::Tensor& high);

std::shared_ptr<AnySpace> make_box_space_any(const torch::Tensor& low, const torch::Tensor& high);

} // namespace spaces
} // namespace lab
