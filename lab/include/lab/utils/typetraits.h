#pragma once

#include "lab/common/common.h"

namespace lab {
namespace spaces {
class Space;
class DiscreteImpl;
class BoxImpl;
} // namespace spaces
namespace agents {
class Algorithm;
}
namespace distributions {
class Distribution;
class Bernoulli;
class Beta;
class Categorical;
class Cauchy;
class Dirichlet;
class ExponentialFamily;
class Normal;
} // namespace distributions

} // namespace lab
namespace lab {
namespace utils {
// Helper to check for the `sample` method
template <typename, typename T, typename = void>
struct has_sample_method : std::false_type {};
template <typename T, typename Ret, typename... Args>
struct has_sample_method<T, Ret(Args...), std::void_t<decltype(std::declval<T>().sample(std::declval<Args>()...))>>
    : std::true_type {};

// Helper to check for the `contains` method
template <typename, typename T, typename = void>
struct has_contains_method : std::false_type {};

template <typename T, typename Ret, typename... Args>
struct has_contains_method<T, Ret(Args...), std::void_t<decltype(std::declval<T>().contains(std::declval<Args>()...))>>
    : std::true_type {};

// Trait to check if a class has both `sample` and `contains` methods
template <typename T>
struct has_sample_and_contains {
  static constexpr bool value =
      has_sample_method<T, torch::Tensor()>::value && has_contains_method<T, bool(torch::Tensor)>::value;
};

template <typename T>
struct has_step {
  using yes = int8_t;
  using no = int16_t;

  template <typename U>
  static yes test_step(decltype(&U::step));
  template <typename U>
  static no test_step(...);
  static constexpr bool value = sizeof(test_step<T>(nullptr)) == sizeof(yes);
};

template <typename T>
using has_step_v = typename has_step<T>::value;

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
struct is_space_holder_of_impl<true, T, C> : std::is_same<typename T::ContainedType, C> {};

template <typename T, typename C>
struct is_space_holder_of : is_space_holder_of_impl<is_space_holder<T>::value, std::decay_t<T>, std::decay_t<C>> {};

template <typename S>
using is_space = std::is_base_of<lab::spaces::Space, typename std::decay<S>::type>;
template <typename S, typename T = void>
using enable_if_space_t = typename std::enable_if<is_space<S>::value, T>::type;
template <typename S>
using is_algorithm = std::is_base_of<lab::agents::Algorithm, typename std::decay<S>::type>;
template <typename S, typename T = void>
using enable_if_algorithm_t = typename std::enable_if<is_algorithm<S>::value, T>::type;
template <typename O>
using is_optim = std::is_base_of<torch::optim::Optimizer, typename std::decay<O>::type>;
template <typename O, typename T = void>
using enable_if_optim_t = typename std::enable_if<is_optim<O>::value, T>::type;
template <typename D>
using is_distribution = std::is_base_of<lab::distributions::Distribution, typename std::decay<D>::type>;
template <typename D, typename T = void>
using enable_if_distribution_t = typename std::enable_if<is_distribution<D>::value, T>::type;

template <typename T, typename... Ts>
struct is_one_of : std::false_type {};
template <typename T, typename First, typename... Ts>
struct is_one_of<T, First, Ts...> : std::conditional_t<std::is_same_v<T, First>, std::true_type, is_one_of<T, Ts...>> {
};
template <typename T, typename... Ts>
inline constexpr bool is_one_of_v = is_one_of<T, Ts...>::value;
template <typename D>
using enable_if_discrete_pd_t =
    typename std::enable_if<is_one_of_v<D, distributions::Bernoulli, distributions::Categorical>, D>::type;
template <typename D>
using enable_if_continuous_pd_t = typename std::enable_if<
    is_one_of_v<D, distributions::Beta, distributions::Cauchy, distributions::Dirichlet, distributions::Normal>,
    D>::type;

template <class... Ts>
struct types_t {};
template <class... Ts>
constexpr types_t<Ts...> types{};
template <class T>
struct tag_t {
  using type = T;
  template <class... Ts>
  constexpr decltype(auto) operator()(Ts&&... ts) const {
    return T{}(std::forward<Ts>(ts)...);
  }
};
template <class T>
constexpr tag_t<T> tag{};

template <template <class...> class Z>
struct template_tag_map {
  template <class In>
  constexpr decltype(auto) operator()(In in_tag) const {
    return tag<Z<typename decltype(in_tag)::type>>;
  }
};

template <class R = void, class Test, class Op, class T0>
R type_switch(Test&&, Op&& op, T0&& t0) {
  return static_cast<R>(op(std::forward<T0>(t0)));
}

template <class R = void, class Test, class Op, class T0, class... Ts>
auto type_switch(Test&& test, Op&& op, T0&& t0, Ts&&... ts) {
  if (test(t0))
    return static_cast<R>(op(std::forward<T0>(t0)));
  return type_switch<R>(test, op, std::forward<Ts>(ts)...);
}

template <class R, class maker_map, class types>
struct named_factory_t;

template <class R, class maker_map, class... Ts>
struct named_factory_t<R, maker_map, types_t<Ts...>> {
  template <class... Args>
  auto operator()(std::string_view sv, Args&&... args) const {
    return type_switch<R>(
        [&sv](auto tag) { return decltype(tag)::type::name == sv; },
        [&](auto tag) { return maker_map{}(tag)(std::forward<Args>(args)...); },
        tag<Ts>...);
  }
};

struct shared_ptr_maker {
  template <class Tag>
  constexpr auto operator()(Tag ttag) {
    using T = typename decltype(ttag)::type;
    return [](auto&&... args) { return std::make_shared<T>(decltype(args)(args)...); };
  }
};

struct object_maker {
  template <class Tag>
  constexpr auto operator()(Tag ttag) {
    using T = typename decltype(ttag)::type;
    return [](auto&&... args) { return T(decltype(args)(args)...); };
  }
};

template <class Second, class First>
struct compose {
  template <class... Args>
  constexpr decltype(auto) operator()(Args&&... args) const {
    return Second{}(First{}(std::forward<Args>(args)...));
  }
};
} // namespace utils
} // namespace lab

#define LAB_FUNC_CALL_MAKER(Func)                                             \
  namespace lab {                                                             \
  namespace utils {                                                           \
  struct Func##_call_maker {                                                  \
    template <class Tag>                                                      \
    constexpr auto operator()(Tag ttag) {                                     \
      using T = typename decltype(ttag)::type;                                \
      return [](auto&&... args) { return T::Func(decltype(args)(args)...); }; \
    }                                                                         \
  };                                                                          \
  }                                                                           \
  }

LAB_FUNC_CALL_MAKER(update)
LAB_FUNC_CALL_MAKER(sample)