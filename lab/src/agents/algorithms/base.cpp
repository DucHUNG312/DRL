#include "lab/agents/algorithms/base.h"
#include "lab/utils/convert.h"
#include "lab/utils/tensor.h"

namespace lab
{

namespace agents
{

Algorithm::Algorithm(const utils::AlgorithmSpec& spec)
    : spec_(std::move(spec))
{}

torch::Tensor Algorithm::calc_pdparam(const torch::Tensor& x)
{
    return net_->forward(x);
}

torch::Tensor Algorithm::calc_pdparam_batch(const Algorithm::ExperienceDict& experiences)
{
    auto batch = experiences.at("state");
    torch::Tensor states = utils::get_tensor_from_ivalue_list(batch);
    if (env_->is_venv())
        states = utils::venv_unpack(states);
    return calc_pdparam(states);
}

void Algorithm::save(torch::serialize::OutputArchive& archive) const
{

}

void Algorithm::load(torch::serialize::InputArchive& archive)
{

}

torch::serialize::OutputArchive& operator<<(torch::serialize::OutputArchive& archive, const std::shared_ptr<Algorithm>& algo)
{
    LAB_CHECK(algo != nullptr);
    algo->save(archive);
    return archive;
}

torch::serialize::InputArchive& operator>>(torch::serialize::InputArchive& archive, const std::shared_ptr<Algorithm>& algo)
{
    LAB_CHECK(algo != nullptr);
    algo->load(archive);
    return archive;
}

}

}