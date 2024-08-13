#pragma once

#include "lab/spaces/base.h"

namespace lab
{
namespace spaces
{

class Discrete : public SpaceBase<int64_t>
{
    LAB_ARG(int64_t, n);
    LAB_ARG(int64_t, start);
public:
    /*For IValue::to<T>*/
    using element_type = lab::spaces::Discrete;

    Discrete();
    Discrete(int64_t n, int64_t start = 0);
    Discrete(const Discrete& dis);
    Discrete(Discrete&& dis) noexcept;
    virtual ~Discrete() override = default;

    virtual int64_t sample() override;

    virtual bool contains(const int64_t& x) const override;
};

LAB_FORCE_INLINE std::ostream& operator<<(std::ostream& out, const Discrete& discrete) 
{
    out << "[";
    out << discrete.n();
    out << ", ";
    out << discrete.start();
    out << "]";
    return out;
}

LAB_FORCE_INLINE Discrete make_discrete_space(int64_t n, int64_t start = 0) 
{
    return Discrete(n , start);
}

LAB_FORCE_INLINE bool operator==(Discrete& d1, Discrete& d2) 
{
  return (d1.n() == d2.n()) && (d1.start() == d2.start());
}

LAB_FORCE_INLINE bool operator!=(Discrete& d1, Discrete& d2) 
{
  return !(d1 == d2);
}

LAB_FORCE_INLINE bool operator==(Discrete& d1, const Discrete& d2) 
{
  return (d1.n() == d2.n()) && (d1.start() == d2.start());
}

LAB_FORCE_INLINE bool operator!=(Discrete& d1, const Discrete& d2) 
{
  return !(d1 == d2);
}

LAB_FORCE_INLINE bool operator==(const Discrete& d1, const Discrete& d2) 
{
  return (d1.n() == d2.n()) && (d1.start() == d2.start());
}

LAB_FORCE_INLINE bool operator!=(const Discrete& d1, const Discrete& d2) 
{
  return !(d1 == d2);
}

}
}


