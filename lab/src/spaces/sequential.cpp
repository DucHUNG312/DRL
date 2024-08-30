#include "lab/spaces/sequential.h"

namespace lab
{
namespace spaces
{

SequentialImpl::SequentialImpl(torch::OrderedDict<std::string, AnySpace>&& ordered_dict) 
{
    spaces_.reserve(ordered_dict.size());
    for (auto& item : ordered_dict) 
        push_back(item.key(), std::move(item.value()));
}

SequentialImpl::SequentialImpl(std::initializer_list<NamedAnySpace> named_spaces) 
{
    spaces_.reserve(named_spaces.size());
    for (const auto& named_space : named_spaces)
        push_back(named_space.name(), named_space.space());
}

std::shared_ptr<Space> SequentialImpl::clone(const std::optional<torch::Device>& device /*= std::nullopt*/) const 
{
    auto clone = std::make_shared<SequentialImpl>();
    for (const auto& space : spaces_)
        clone->push_back(space.clone(device));
    return clone;
}

void SequentialImpl::reset()
{
    LAB_LOG_WARN("reset() is empty for Sequential, since it does not have parameters of its own.");
    LAB_UNREACHABLE;
}

void SequentialImpl::pretty_print(std::ostream& stream) const 
{
    stream << "lab::spaces::Sequential";
}

std::vector<torch::Tensor> SequentialImpl::sample(/*std::vector<torch::Tensor>&& inputs*/) 
{
    LAB_CHECK(!is_empty());
    //LAB_CHECK(inputs.size() == spaces_.size());

    std::vector<torch::Tensor> values;
    values.reserve(spaces_.size());

    for (int64_t i = 0; i < spaces_.size(); i++)
        values.push_back(spaces_[i].sample(/*std::forward<torch::Tensor>(inputs[i])*/));

    return values;
}

void SequentialImpl::push_back(AnySpace any_space) 
{
    push_back(c10::to_string(spaces_.size()), std::move(any_space));
}

void SequentialImpl::push_back(std::string name, AnySpace any_space) 
{
    spaces_.push_back(std::move(any_space));
    const auto index = spaces_.size() - 1;
    register_subspace(std::move(name), spaces_[index].ptr());
}

SequentialImpl::Iterator SequentialImpl::begin() 
{
    return spaces_.begin();
}

SequentialImpl::ConstIterator SequentialImpl::begin() const 
{
    return spaces_.begin();
}

SequentialImpl::Iterator SequentialImpl::end() 
{
    return spaces_.end();
}

SequentialImpl::ConstIterator SequentialImpl::end() const 
{
    return spaces_.end();
}

std::shared_ptr<Space> SequentialImpl::ptr(size_t index) const 
{
    LAB_CHECK(index < size());
    return spaces_[index].ptr();
}

std::shared_ptr<Space> SequentialImpl::operator[](size_t index) const 
{
    return ptr(index);
}

size_t SequentialImpl::size() const noexcept 
{
    return spaces_.size();
}

bool SequentialImpl::is_empty() const noexcept 
{
    return size() == 0;
}


Sequential::Sequential() 
    : SpaceHolder<SequentialImpl>() {}

Sequential::Sequential(std::initializer_list<NamedAnySpace> named_spaces)
    : SpaceHolder<SequentialImpl>(std::make_shared<SequentialImpl>(named_spaces)) {}
}

}