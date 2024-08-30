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

void Memory::pretty_print(std::ostream& stream, const std::string& indentation) const
{
    const std::string next_indentation = indentation + "  ";
    stream << "Memory(\n";
    pretty_print_list(stream, "state", next_indentation, [](const torch::IValue& value) { return value.toTensor(); });
    pretty_print_list(stream, "next_state", next_indentation, [](const torch::IValue& value) { return value.toTensor(); });
    pretty_print_list(stream, "action", next_indentation, [](const torch::IValue& value) { return value.toTensor(); });
    pretty_print_list(stream, "reward", next_indentation, [](const torch::IValue& value) { return value.toDouble(); });
    pretty_print_list(stream, "terminated", next_indentation, [](const torch::IValue& value) { return value.toBool(); });
    pretty_print_list(stream, "truncated", next_indentation, [](const torch::IValue& value) { return value.toBool(); });
    stream << ")";
}

std::ostream& Memory::operator<<(std::ostream& stream)
{
    pretty_print(stream, "");
    return stream;
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