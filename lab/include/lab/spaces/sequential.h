#pragma once

#include "lab/spaces/base.h"
#include "lab/spaces/any.h"

namespace lab
{
namespace spaces
{

// The same as torch::nn::Sequential
class SequentialImpl : public ClonableSpace<SequentialImpl>
{
public:
  using Iterator = std::vector<AnySpace>::iterator;
  using ConstIterator = std::vector<AnySpace>::const_iterator;

  SequentialImpl() = default;

  template <typename... Spaces>
  explicit SequentialImpl(Spaces&&... spaces);

  explicit SequentialImpl(torch::OrderedDict<std::string, AnySpace>&& ordered_dict);

  explicit SequentialImpl(std::initializer_list<NamedAnySpace> named_spaces);

  std::shared_ptr<Space> clone(const std::optional<torch::Device>& device = std::nullopt) const override;

  void reset() override;

  void pretty_print(std::ostream& stream) const override;

  std::vector<torch::Tensor> sample(/*std::vector<torch::Tensor>&& inputs*/);

  template <typename... InputTypes>
  std::vector<bool> contains(InputTypes&&... inputs);

  template <typename SpaceType>
  void push_back(std::shared_ptr<SpaceType> space_ptr);

  template <typename SpaceType>
  void push_back(std::string name, std::shared_ptr<SpaceType> space_ptr);

  template <typename S, typename = utils::enable_if_space_t<S>>
  void push_back(S&& space);

  template <typename S, typename = utils::enable_if_space_t<S>>
  void push_back(std::string name, S&& space);

  template <typename S>
  void push_back(const SpaceHolder<S>& space_holder);

  template <typename S>
  void push_back(std::string name, const SpaceHolder<S>& space_holder);

  template <typename Container>
  void extend(const Container& container);

  void push_back(AnySpace any_space);

  void push_back(std::string name, AnySpace any_space);

  Iterator begin();

  ConstIterator begin() const;

  Iterator end();

  ConstIterator end() const;

  template <typename T>
  T& at(size_t index);

  template <typename T>
  const T& at(size_t index) const;

  std::shared_ptr<Space> ptr(size_t index) const;

  template <typename T>
  std::shared_ptr<T> ptr(size_t index) const;

  std::shared_ptr<Space> operator[](size_t index) const;

  size_t size() const noexcept;

  bool is_empty() const noexcept;
private:
  template <typename First, typename Second, typename... Rest,
      typename = std::enable_if_t<!std::is_same_v<First, std::string> && !std::is_same_v<std::decay_t<First>, std::decay_t<const char (&)[]>>>>
  void push_back(First&& first, Second&& second, Rest&&... rest);

  void push_back() {}

  template <size_t I, typename First, typename... InputTypes>
  std::vector<bool> contains_impl(First&& first, InputTypes&&... inputs);

  template <size_t I>
  std::vector<bool> contains_impl();
private:
  std::vector<AnySpace> spaces_;
};

template <typename... Spaces>
SequentialImpl::SequentialImpl(Spaces&&... spaces) 
{
    spaces_.reserve(sizeof...(Spaces));
    push_back(std::forward<Spaces>(spaces)...);
}

template <typename... InputTypes>
std::vector<bool> SequentialImpl::contains(InputTypes&&... inputs) 
{
  LAB_CHECK(!is_empty());
  LAB_CHECK(sizeof...(InputTypes) == spaces_.size());
  return contains_impl<0, InputTypes...>(std::forward<InputTypes>(inputs)...);
}

template <typename SpaceType>
void SequentialImpl::push_back(std::shared_ptr<SpaceType> space_ptr) 
{
  push_back(c10::to_string(spaces_.size()), std::move(space_ptr));
}

template <typename SpaceType>
void SequentialImpl::push_back(std::string name, std::shared_ptr<SpaceType> space_ptr) 
{
  push_back(std::move(name), AnySpace(std::move(space_ptr)));
}

template <typename S, typename>
void SequentialImpl::push_back(S&& space) 
{
  push_back(c10::to_string(spaces_.size()), std::forward<S>(space));
}

template <typename S, typename>
void SequentialImpl::push_back(std::string name, S&& space) 
{
  using Type = typename std::remove_reference_t<S>;
  push_back(std::move(name), std::make_shared<Type>(std::forward<S>(space)));
}

template <typename S>
void SequentialImpl::push_back(const SpaceHolder<S>& space_holder) 
{
  push_back(c10::to_string(spaces_.size()), space_holder);
}

/// Unwraps the contained named space of a `SpaceHolder` and adds it to the
/// `Sequential`.
template <typename S>
void SequentialImpl::push_back(std::string name, const SpaceHolder<S>& space_holder) 
{
  push_back(std::move(name), space_holder.ptr());
}

/// Iterates over the container and calls `push_back()` on each value.
template <typename Container>
void SequentialImpl::extend(const Container& container) 
{
  for (const auto& space : container)
    push_back(space);
}

template <typename T>
T& SequentialImpl::at(size_t index) 
{
  static_assert(
      utils::is_space<T>::value,
      "Can only call Sequential::at with an spaces::Space type");
  LAB_CHECK(index < size());
  return spaces_[index].get<T>();
}

template <typename T>
const T& SequentialImpl::at(size_t index) const 
{
  static_assert(
      utils::is_space<T>::value,
      "Can only call Sequential::at with an spaces::Space type");
  LAB_CHECK(index < size());
  return spaces_[index].get<T>();
}

template <typename T>
std::shared_ptr<T> SequentialImpl::ptr(size_t index) const 
{
  static_assert(
      utils::is_space<T>::value,
      "Can only call Sequential::ptr with an spaces::Space type");
  LAB_CHECK(index < size());
  return spaces_[index].ptr<T>();
}

template <typename First, typename Second, typename... Rest, typename>
void SequentialImpl::push_back(First&& first, Second&& second, Rest&&... rest) 
{
  push_back(std::forward<First>(first));
  push_back(std::forward<Second>(second), std::forward<Rest>(rest)...);
}

template <size_t I, typename First, typename... InputTypes>
std::vector<bool> SequentialImpl::contains_impl(First&& first, InputTypes&&... inputs) 
{
  static_assert(I < sizeof...(InputTypes) + 1, "Index out of bounds");

  bool result = spaces_[I].contains(std::forward<First>(first));

  std::vector<bool> results = {result};

  if constexpr (sizeof...(InputTypes) > 0) 
  {
    auto rest_results = contains_impl<I + 1>(std::forward<InputTypes>(inputs)...);
    results.insert(results.end(), rest_results.begin(), rest_results.end());
  }

  return results;
}

template <size_t I>
std::vector<bool> SequentialImpl::contains_impl() 
{
  return {}; 
}

class Sequential : public SpaceHolder<SequentialImpl> 
{
public:
  using SpaceHolder<SequentialImpl>::SpaceHolder;

  Sequential();
  Sequential(std::initializer_list<NamedAnySpace> named_spaces);
};

}
}

