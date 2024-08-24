#include "lab/spaces/discrete.h"

namespace lab
{
namespace spaces
{

DiscreteOptions::DiscreteOptions(int64_t n, int64_t start)
    : n_(n), start_(start) {}

DiscreteImpl::DiscreteImpl(const DiscreteOptions& options_)
    : options(options_)
{
    reset();
}

void DiscreteImpl::reset()
{
    n = register_parameter("n", torch::tensor({options.n()}, torch::kInt64));
    start = register_parameter("start", torch::tensor({options.start()}, torch::kInt64));
    shape_ = torch::tensor({1}, torch::kInt64);
    name_ = "Discrete";
}

void DiscreteImpl::pretty_print(std::ostream& stream) const
{
    stream << std::boolalpha
    << "lab::spaces::Discrete(n=" << options.n()
    << ", start=" << options.start() << ")";
}

int64_t DiscreteImpl::sample(/*const torch::Tensor& mask*/)
{
    // if(mask.defined())
    // {
    //     LAB_CHECK(mask.dim() == 1 && mask.sizes() == n.sizes());
    //     LAB_CHECK(!utils::has_all_zeros(mask));
    //     torch::Tensor valid_indices = torch::nonzero(mask).to(torch::kInt64);
    //     torch::Tensor random_index = start + valid_indices[rand_.sample_int_uniform(0, valid_indices.size(0))];
    //     return random_index.item<int64_t>();
    // }

    return rand_.sample_int_uniform(options.start(), options.start() + options.n());
}

bool DiscreteImpl::contains(int64_t x) const
{
    return (options.start() <= x) && (x < options.start() + options.n());
}

}
}