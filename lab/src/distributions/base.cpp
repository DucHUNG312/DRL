#include "lab/distributions/base.h"

namespace lab
{
namespace distributions
{

Distribution::Distribution(torch::IntArrayRef batch_shape)
    : batch_shape_(batch_shape) {}

std::vector<int64_t> Distribution::extended_shape(torch::IntArrayRef sample_shape /*= {}*/)
{
    std::vector<int64_t> combined_shape;
    if(!sample_shape.empty())
        combined_shape.insert(combined_shape.end(), sample_shape.begin(), sample_shape.end());
    combined_shape.insert(combined_shape.end(), batch_shape_.begin(), batch_shape_.end());
    return combined_shape;
}

}
}