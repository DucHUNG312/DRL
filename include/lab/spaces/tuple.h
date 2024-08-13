#pragma once

#include "lab/spaces/base.h"

namespace lab
{
namespace spaces
{
namespace experiment
{

class Tuple : public SpaceBase<c10::intrusive_ptr<c10::ivalue::Tuple>>
{
    LAB_ARG(c10::intrusive_ptr<c10::ivalue::Tuple>, spaces);
public:
    /*For IValue::to<T>*/
    using element_type = lab::spaces::experiment::Tuple;

    Tuple(const std::vector<c10::IValue>& spaces);
    Tuple(const Tuple& tup);
    Tuple(Tuple&& tup)noexcept;

    virtual ~Tuple() override = default;

    virtual c10::intrusive_ptr<c10::ivalue::Tuple> sample() override;

    virtual bool contains(const c10::intrusive_ptr<c10::ivalue::Tuple>& x) const override;

    int64_t size() const;

    c10::ivalue::TupleElements get_elements() const;  
};

LAB_FORCE_INLINE std::ostream& operator<<(std::ostream& out, const Tuple& tup) 
{
    out << "[ ";
    for (auto& space : tup.get_elements())
    {
        out << space;
        out << " ";
    }
    out << "]";
    return out;
}

LAB_FORCE_INLINE Tuple make_tuple_space(std::vector<c10::IValue>& spaces) 
{
    return Tuple(spaces);
}

LAB_FORCE_INLINE bool operator==(Tuple& t1, Tuple& t2) 
{
  return (t1.size() == t2.size()) && t1.contains(t2.spaces());
}

LAB_FORCE_INLINE bool operator!=(Tuple& t1, Tuple& t2) 
{
  return !(t1 == t2);
}

LAB_FORCE_INLINE bool operator==(Tuple& t1, const Tuple& t2) 
{
  return (t1.size() == t2.size()) && t1.contains(t2.spaces());
}

LAB_FORCE_INLINE bool operator!=(Tuple& t1, const Tuple& t2) 
{
  return !(t1 == t2);
}

LAB_FORCE_INLINE bool operator==(const Tuple& t1, const Tuple& t2) 
{
  return (t1.size() == t2.size()) && t1.contains(t2.spaces());
}

LAB_FORCE_INLINE bool operator!=(const Tuple& t1, const Tuple& t2) 
{
  return !(t1 == t2);
}

}
}
}

