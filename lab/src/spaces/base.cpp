#include "lab/spaces/base.h"

namespace lab
{

namespace spaces
{

std::shared_ptr<Space> Space::clone(const std::optional<torch::Device>& device /*= std::nullopt*/) const
{
    LAB_LOG_FATAL("Cannot call clone() on Space top class.");
    return nullptr;
}

void Space::clone_(Space& other, const std::optional<torch::Device>& device)
{
    LAB_LOG_FATAL("Cannot call clone_() on Space top class.");
}

bool Space::is_serializable() const
{
    return true;
}

void Space::pretty_print_recursive(std::ostream& stream, const std::string& indentation) const
{
    pretty_print(stream);
    if (!children_.is_empty()) 
    {
        stream << "(\n";
        const std::string next_indentation = indentation + "  ";
        for (const auto& child : children_) 
        {
            stream << next_indentation << "(" << child.key() << "): ";
            child.value()->pretty_print_recursive(stream, next_indentation);
            stream << '\n';
        }
        stream << indentation << ")";
    }
}

void Space::pretty_print(std::ostream& stream) const
{
    stream << name_;
}

std::ostream& Space::operator<<(std::ostream& stream)
{
    pretty_print_recursive(stream, "");
    return stream;
}

void Space::save(torch::serialize::OutputArchive& archive) const 
{
    archive.write("shape", shape_);
    archive.write("name", name_);

    for (const auto& parameter : parameters_)
        archive.write(parameter.key(), parameter.value());

    for (const auto& child : children_) 
    {
        if (child.value()->is_serializable()) 
        {
            torch::serialize::OutputArchive child_archive(archive.compilation_unit());
            child.value()->save(child_archive);
            archive.write(child.key(), child_archive);
        }
    }
}

void Space::load(torch::serialize::InputArchive& archive) 
{
    archive.read("shape", shape_);

    torch::IValue name;
    archive.read("name", name);
    name_ = name.toString();

    for (auto& parameter : parameters_)
        archive.read(parameter.key(), parameter.value());
    
    for (const auto& child : children_) 
    {
        if (child.value()->is_serializable()) 
        {
            torch::serialize::InputArchive child_archive;
            archive.read(child.key(), child_archive);
            child.value()->load(child_archive);
        }
    }
}

torch::Tensor& Space::register_parameter(std::string name, torch::Tensor tensor, bool requires_grad /*= false*/)
{
    LAB_CHECK(!name.empty());
    LAB_CHECK(name.find('.') == std::string::npos);
    if (!tensor.defined()) 
        if (requires_grad)
            LAB_LOG_WARN("An undefined tensor cannot require grad. Ignoring the `requires_grad=true` function parameter.");
    else
        tensor.set_requires_grad(requires_grad);
    return parameters_.insert(std::move(name), std::move(tensor));
}

torch::Tensor& Space::get_parameter(std::string name)
{
    LAB_CHECK(!name.empty());
    LAB_CHECK(name.find('.') == std::string::npos);
    LAB_CHECK(!parameters_.is_empty());

    if(parameters_.contains(name))
        return parameters_[name];
    throw std::runtime_error("No key with name " + name + " exists in parameters!");
}

Space::Iterator Space::begin() 
{
    return children_.begin();
}

Space::ConstIterator Space::begin() const 
{
    return children_.begin();
}

Space::Iterator Space::end() 
{
    return children_.end();
}

Space::ConstIterator Space::end() const 
{
    return children_.end();
}

torch::serialize::OutputArchive& operator<<(torch::serialize::OutputArchive& archive, const std::shared_ptr<Space>& space)
{
    LAB_CHECK(space != nullptr);
    space->save(archive);
    return archive;
}

torch::serialize::InputArchive& operator>>(torch::serialize::InputArchive& archive, const std::shared_ptr<Space>& space)
{
    LAB_CHECK(space != nullptr);
    space->load(archive);
    return archive;
}

}

}