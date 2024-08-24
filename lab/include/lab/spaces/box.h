#pragma once

#include "lab/spaces/base.h"

namespace lab
{
namespace spaces
{

struct BoxOptions 
{
  BoxOptions(const torch::Tensor& low, const torch::Tensor& high);
  LAB_ARG(torch::Tensor, low);
  LAB_ARG(torch::Tensor, high);
};

class BoxImpl : public ClonableSpace<BoxImpl>
{
public:
    BoxImpl() = default;
    explicit BoxImpl(const BoxOptions& options_);
    explicit BoxImpl(const torch::Tensor& low, const torch::Tensor& high)
      : BoxImpl(BoxOptions(low, high)) {}

    void reset() override;

    torch::Tensor sample();

    bool contains(const torch::Tensor& x) const;
public:
  BoxOptions options;
  torch::Tensor low;
  torch::Tensor high;
};

LAB_SPACE(Box);

LAB_FORCE_INLINE Box make_box_space(torch::IntArrayRef shape = torch::IntArrayRef(1)) 
{
  return static_cast<Box>(std::make_shared<BoxImpl>(torch::full(shape, 0, torch::kDouble), torch::full(shape, 1, torch::kDouble)));
}

LAB_FORCE_INLINE Box make_box_space(const torch::Tensor& low, const torch::Tensor& high) 
{
  return static_cast<Box>(std::make_shared<BoxImpl>(low, high));
}

}
}

