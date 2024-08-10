#include "lab/utils/rand.h"

namespace lab
{
namespace utils
{
torch::Generator Rand::generator;

Rand::Rand(uint64_t seed /* = 42 */)
{
    seed = seed;
    generator = at::detail::createCPUGenerator(seed);
}

void Rand::set_seed(uint64_t seed)
{
    seed = seed;
    generator.set_current_seed(seed);
}

int64_t Rand::choice(const torch::Tensor& prob, double u /* = -1 */)
{
    LAB_CHECK_EQ(prob.dim(), 1);
    int64_t n = prob.size(0);
    if (n == 0)
    {
        LAB_LOG_WARN("Calling Rand::choice with empty probability tensor! Returning -1.");
        return -1;
    }
    
    torch::Tensor cumul = torch::cumsum(prob, 0);

    // Get sample 
    double unif_sample;
    if (u == -1) unif_sample = torch::rand({1}, Rand::generator, torch::TensorOptions().dtype(torch::kDouble)).item<double>(); 
    else  unif_sample = u;

    torch::Tensor mask = torch::ge(cumul, unif_sample);
    torch::Tensor indices = torch::nonzero(mask).to(torch::kInt64);

    return (indices.size(0) > 0) ? indices[0].item<int64_t>() : -1;
}

double Rand::sample_real_uniform(double a, double b)
{
    LAB_CHECK_GE(b, a);
    auto uniform_tensor = torch::rand({1}, Rand::generator, torch::TensorOptions().dtype(torch::kDouble));
    double uniform_value = uniform_tensor.item<double>();
    return uniform_value * (b - a) + a;
}

torch::Tensor Rand::sample_real_uniform(double a, double b, const torch::IntArrayRef& shape)
{
    LAB_CHECK_GE(b, a);
    auto uniform_tensor = torch::rand(shape, Rand::generator, torch::TensorOptions().dtype(torch::kDouble));
    return uniform_tensor * (b - a) + a;
}

int64_t Rand::sample_int_uniform(int64_t a, int64_t b)
{
    auto uniform_tensor = torch::randint(a, b, {1}, Rand::generator, torch::TensorOptions().dtype(torch::kInt64));
    return uniform_tensor.item<int64_t>();
}

torch::Tensor Rand::sample_int_uniform(int64_t a, int64_t b, const torch::IntArrayRef& shape)
{
    auto uniform_tensor = torch::randint(a, b, shape, Rand::generator, torch::TensorOptions().dtype(torch::kInt64));
    return uniform_tensor;
}

double Rand::sample_gaussian(double mu, double sigma)
{
    LAB_CHECK_GT(sigma, 0);
    auto normal_tensor = torch::normal(mu, sigma, {1}, Rand::generator, torch::TensorOptions().dtype(torch::kDouble));
    return normal_tensor.item<double>();
}

torch::Tensor Rand::sample_gaussian(double mu, double sigma, const torch::IntArrayRef& shape)
{
    LAB_CHECK_GT(sigma, 0);
    auto normal_tensor = torch::normal(mu, sigma, shape, Rand::generator, torch::TensorOptions().dtype(torch::kDouble));
    return normal_tensor;
}

} 
} 