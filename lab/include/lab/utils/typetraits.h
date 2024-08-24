#pragma once

#include "lab/core.h"

namespace lab
{
namespace spaces
{
class Space;
}
}

namespace lab
{

namespace utils
{

template <typename T>
struct has_sample_and_contains
{
  using yes = int8_t;
  using no = int16_t;

  template <typename U>
  static yes test_sample(decltype(&U::sample));
  template <typename U>
  static yes test_contains(decltype(&U::contains));
  template <typename U>
  static no test_sample(...);
  template <typename U>
  static no test_contains(...);

  static constexpr bool value = (
    sizeof(test_sample<T>(nullptr)) == sizeof(yes) && 
    sizeof(test_contains<T>(nullptr)) == sizeof(yes));
};

struct SpaceHolderIndicator {};

template <typename T>
using is_space_holder = std::is_base_of<SpaceHolderIndicator, std::decay_t<T>>;

template <typename T>
using disable_if_space_holder_t = std::enable_if_t<!is_space_holder<T>::value>;

template <bool is_space_holder_value, typename T, typename C>
struct is_space_holder_of_impl;

template <typename T, typename C>
struct is_space_holder_of_impl<false, T, C> : std::false_type {};

template <typename T, typename C>
struct is_space_holder_of_impl<true, T, C>
    : std::is_same<typename T::ContainedType, C> {};

template <typename T, typename C>
struct is_space_holder_of : is_space_holder_of_impl<
                                 is_space_holder<T>::value,
                                 std::decay_t<T>,
                                 std::decay_t<C>> {};

template <typename S>
using is_space = std::is_base_of<lab::spaces::Space, typename std::decay<S>::type>;

template <typename S, typename T = void>
using enable_if_space_t = typename std::enable_if<is_space<S>::value, T>::type;

}

}