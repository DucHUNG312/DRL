#pragma once

#include "lab/spaces/base.h"

namespace lab
{
namespace spaces
{

class Box : public SpaceBase<torch::Tensor>
{
    LAB_ARG(torch::Tensor, low);
    LAB_ARG(torch::Tensor, high);
public:
    /*For IValue::to<T>*/
    using element_type = lab::spaces::Box;

    Box();
    Box(
        const torch::Tensor& low, 
        const torch::Tensor& high
    );

    Box(const Box& box) = default;
    Box& operator=(const Box& other) = default;
    Box(Box&& box) noexcept = default;
    Box& operator=(Box&& other) noexcept = default;
    virtual ~Box() = default;

    virtual torch::Tensor sample() override;

    virtual bool contains(const torch::Tensor& x) const override;
};

LAB_FORCE_INLINE std::ostream& operator<<(std::ostream& out, const Box& box) 
{
    out << "[";
    out << box.low();
    out << ", ";
    out << box.high();
    out << "]";
    return out;
}

LAB_FORCE_INLINE Box make_box_space(torch::IntArrayRef shape = torch::IntArrayRef(1)) 
{
  return Box(torch::full(shape, 0, torch::kDouble), torch::full(shape, 1, torch::kDouble));
}

LAB_FORCE_INLINE Box make_box_space(torch::Tensor& low, torch::Tensor& high) 
{
  return Box(low, high);
}

LAB_FORCE_INLINE bool operator==(Box& b1, Box& b2) 
{
  return utils::tensor_eq(b1.low(), b2.low()) && utils::tensor_eq(b1.high(), b2.high());
}

LAB_FORCE_INLINE bool operator!=(Box& b1, Box& b2) 
{
  return !(b1 == b2);
}

LAB_FORCE_INLINE bool operator==(Box& b1, const Box& b2) 
{
  return utils::tensor_eq(b1.low(), b2.low()) && utils::tensor_eq(b1.high(), b2.high());
}

LAB_FORCE_INLINE bool operator!=(Box& b1, const Box& b2) 
{
  return !(b1 == b2);
}

LAB_FORCE_INLINE bool operator==(const Box& b1, const Box& b2) 
{
  return utils::tensor_eq(b1.low(), b2.low()) && utils::tensor_eq(b1.high(), b2.high());
}

LAB_FORCE_INLINE bool operator!=(const Box& b1, const Box& b2) 
{
  return !(b1 == b2);
}
}
}

