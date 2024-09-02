#include "lab/utils/rand.h"

namespace lab
{
namespace utils
{

double Rand::rand()
{
    auto uniform_tensor = torch::rand({1}, torch::TensorOptions().dtype(torch::kDouble));
    return uniform_tensor.item<double>();
}

Rand::Rand(uint64_t seed /* = 0 */)
{
    generator_ = torch::make_generator<torch::CPUGeneratorImpl>(seed);
    reset(seed);
}

void Rand::reset(uint64_t seed /* = 0 */)
{
    set_current_seed(seed);
}

void Rand::set_current_seed(uint64_t seed)
{
    generator_.set_current_seed(seed);
}

uint64_t Rand::current_seed()
{
    return generator_.current_seed();
}

void Rand::set_state(const torch::Tensor& new_state)
{
    generator_.set_state(new_state);
}

torch::Tensor Rand::get_state() const
{
    return generator_.get_state();
}

int64_t Rand::choice(const torch::Tensor& prob, std::optional<double> u /*= std::nullopt*/)
{
    LAB_CHECK_EQ(prob.dim(), 1);
    if (prob.size(0) == 0)
    {
        LAB_LOG_WARN("Calling Rand::choice with empty probability tensor! Returning -1.");
        return -1;
    }
    torch::Tensor cumul = torch::cumsum(prob, 0);
    double unif_sample = u.has_value() ? u.value() : sample_uniform(
        torch::tensor({0}, torch::kDouble), 
        torch::tensor({1}, torch::kDouble)).item<double>();
    torch::Tensor mask = torch::ge(cumul, unif_sample);
    torch::Tensor indices = torch::nonzero(mask).to(torch::kInt64);
    return (indices.size(0) > 0) ? indices[0].item<int64_t>() : -1;
}

torch::Tensor Rand::sample_uniform(const torch::Tensor& a, const torch::Tensor& b)
{
    LAB_CHECK_EQ(b.sizes(), a.sizes());
    auto uniform_tensor = torch::rand(a.sizes(), generator_, torch::TensorOptions().dtype(torch::kDouble));
    return (uniform_tensor * (b - a)) + a;
}

torch::Tensor Rand::sample_gaussian(double mu, double sigma, torch::IntArrayRef shape)
{
    LAB_CHECK_GT(sigma, 0);
    auto normal_tensor = torch::normal(mu, sigma, shape, generator_, torch::TensorOptions().dtype(torch::kDouble));
    return normal_tensor;
}

torch::Tensor Rand::gen_int_permutation(int64_t n)
{
    torch::Tensor permutation = torch::randperm(n, generator_, torch::TensorOptions().dtype(torch::kInt64));
    return permutation;
}

int64_t Rand::sample_int_uniform(int64_t a, int64_t b)
{
    LAB_CHECK_LT(a, b);
    auto uniform_tensor = torch::randint(a, b, {1}, generator_, torch::TensorOptions().dtype(torch::kInt64));
    return uniform_tensor.item<int64_t>();
}

double Rand::sample_double_uniform(double a, double b)
{
    LAB_CHECK_LT(a, b);
    auto uniform_tensor = torch::rand({1}, generator_, torch::TensorOptions().dtype(torch::kDouble));
    return uniform_tensor.item<double>() * (b - a) + a;
}

} 
} 