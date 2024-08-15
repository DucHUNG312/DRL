#include "lab/spaces/box.h"

namespace lab
{
namespace spaces
{
    Box::Box() 
    {
        name_ = SpaceType::BOX;
        shape_ = utils::make_shape(int64_t(1));
        low_ = torch::full(shape().to_torch(), 0, torch::kDouble);
        high_ = torch::full(shape().to_torch(), 1, torch::kDouble); 
    }

    Box::Box(
        const torch::Tensor& low, 
        const torch::Tensor& high)
    {
        LAB_CHECK(utils::tensor_lt(low, high));
        name_ = SpaceType::BOX;
        shape_ = utils::make_shape(low.sizes());
        low_ = low.to(torch::kDouble).clone();
        high_ = high.to(torch::kDouble).clone();     
    }

    Box::Box(const Box& box)
        : SpaceBase(box), low_(box.low_.clone()), high_(box.high_.clone()) {}

    Box::Box(Box&& box) noexcept
        : SpaceBase(std::move(box)), low_(std::move(box.low_)), high_(std::move(box.high_)) {}

    Box& Box::operator=(const Box& other) 
    {
        if (this != &other) 
        {
            SpaceBase::operator=(other);
            low_ = other.low_;
            high_ = other.high_;
        }
        return *this;
    }

    Box& Box::operator=(Box&& other) noexcept 
    {
        if (this != &other) 
        {
            SpaceBase::operator=(std::move(other));
            low_ = std::move(other.low_);
            high_ = std::move(other.high_);
        }
        return *this;
    }

    torch::Tensor Box::sample()
    {
        torch::Tensor sample = rand().sample_real_uniform(low_, high_);
        return sample.clone();
    }

    bool Box::contains(const torch::Tensor& x) const
    {
        return (utils::tensor_ge(x, low_) && utils::tensor_le(x, high_));
    }
}
}