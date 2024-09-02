#include "lab/agents/memory/base.h"

namespace lab
{

namespace agents
{

Memory::Memory(const utils::MemorySpec& spec)
    : spec_(std::move(spec))
{
    reset();
}

void Memory::reset()
{
    size_ = 0;
    ready_ = 0;
    most_recent_ = utils::StepResult();
    experiences_.clear();
    for (const auto& key : keys_)
        experiences_.insert(key, c10::impl::GenericList(c10::AnyType::get()));     
}

torch::List<torch::IValue> Memory::get_experiences(const std::string& key) const
{
    return experiences_.at(key);
}

void Memory::save(torch::serialize::OutputArchive& archive) const
{

}

void Memory::load(torch::serialize::InputArchive& archive)
{

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
    LAB_CHECK(memory != nullptr);
    memory->save(archive);
    return archive;
}

torch::serialize::InputArchive& operator>>(torch::serialize::InputArchive& archive, const std::shared_ptr<Memory>& memory)
{
    LAB_CHECK(memory != nullptr);
    memory->load(archive);
    return archive;
}

}

}