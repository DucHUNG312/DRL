#include "lab/agents/memory/base.h"

namespace lab
{

namespace agents
{

Memory::Memory(const std::shared_ptr<Body>& body, const utils::MemorySpec& spec)
    : body_(body), spec_(std::move(spec))
{
    reset();
}

void Memory::reset()
{
    LAB_UNIMPLEMENTED;    
}

void Memory::update(const envs::StepResult& result)
{
    LAB_UNIMPLEMENTED;
}

Memory::ExperienceDict Memory::sample()
{
    LAB_UNIMPLEMENTED;
    return Memory::ExperienceDict();
}

void Memory::save(torch::serialize::OutputArchive& archive) const
{

}

void Memory::load(torch::serialize::InputArchive& archive)
{

}
void Memory::add_experience(const envs::StepResult& result)
{
    LAB_UNIMPLEMENTED;
}

torch::serialize::OutputArchive& operator<<(torch::serialize::OutputArchive& archive, const std::shared_ptr<Memory>& memory)
{
    return archive;
}

torch::serialize::InputArchive& operator>>(torch::serialize::InputArchive& archive, const std::shared_ptr<Memory>& memory)
{
    return archive;
}

}

}