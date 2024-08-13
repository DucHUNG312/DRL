#include "lab/spaces/tuple.h"
#include "lab/spaces/discrete.h"
#include "lab/spaces/box.h"

namespace lab
{
namespace spaces
{
namespace experiment
{

Tuple::Tuple(const std::vector<c10::IValue>& spaces_vec)
{
    name_ = SpaceType::TUPLE;
    spaces_ = c10::ivalue::Tuple::create(spaces_vec);
    std::vector<utils::IShape> shapes;
    for (auto& space : spaces_->elements())
    {
        std::string space_name = utils::get_object_name(space);
        if (space_name == "lab.Discrete")
        {
            auto discrete = space.toCustomClass<lab::spaces::Discrete>();
            shapes.push_back(utils::make_shape(discrete->shape()));
        }
        else if (space_name == "lab.Box")
        {
            auto box = space.toCustomClass<lab::spaces::Box>();
            shapes.push_back(utils::make_shape(box->shape()));
        }
    }
    shape_ = utils::get_bounding_shape(shapes);
}

Tuple::Tuple(const Tuple& tup)
    : SpaceBase(tup), spaces_(tup.spaces_) {}

Tuple::Tuple(Tuple&& tup) noexcept
    : SpaceBase(std::move(tup)), spaces_(std::move(tup.spaces_)) {}

c10::intrusive_ptr<c10::ivalue::Tuple> Tuple::sample()
{
    std::vector<c10::IValue> values;
    for (auto& space : spaces_->elements())
    {
        std::string space_name = utils::get_object_name(space);
        if (space_name == "lab.Discrete")
        {
            auto discrete = space.toCustomClass<lab::spaces::Discrete>();
            values.push_back(discrete->sample());
        }
        else if (space_name == "lab.Box")
        {
            auto box = space.toCustomClass<lab::spaces::Box>();
            values.push_back(box->sample());
        }
    }
    return c10::ivalue::Tuple::create(values);
}

bool Tuple::contains(const c10::intrusive_ptr<c10::ivalue::Tuple>& x) const
{
    auto x_elements = x->elements();
    auto space_elements = spaces_->elements();

    if (x->size() > size()) return false;

    for (auto& x_element : x_elements) 
    {
        bool found = false;
        for (auto& space : space_elements) 
        {
            if (x_element == space) 
            {
                found = true;
                break;
            }
        }
        if (!found)
            return false;
    }
    return true;
}

int64_t Tuple::size() const 
{ 
    return spaces_->size(); 
}

c10::ivalue::TupleElements Tuple::get_elements() const
{
    return spaces_->elements();
}

}
}
}