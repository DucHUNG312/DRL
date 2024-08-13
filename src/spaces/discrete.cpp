#include "lab/spaces/discrete.h"

namespace lab
{
namespace spaces
{

Discrete::Discrete()
{
    name_ = SpaceType::DISCRETE; 
    shape_ = utils::make_shape(int64_t(1)); 
    n_ = 0; 
    start_ = 0; 
}

Discrete::Discrete(int64_t _n, int64_t _start /*= 0*/)
{
    name_ = SpaceType::DISCRETE; 
    shape_ = utils::make_shape(int64_t(1)); 
    n_ = _n;
    start_ = _start;
}

Discrete::Discrete(const Discrete& dis)
    : SpaceBase(dis), n_(dis.n_), start_(dis.start_) {}

Discrete::Discrete(Discrete&& dis) noexcept
    : SpaceBase(std::move(dis)), n_(std::move(dis.n_)), start_(std::move(dis.start_))
{}

int64_t Discrete::sample()
{
    int64_t offset = rand().sample_int_uniform(0, n_);
    return start_ + offset;
}

bool Discrete::contains(const int64_t& x) const
{
    return (x >= start_ && x < start_ + n_);
}

}
}