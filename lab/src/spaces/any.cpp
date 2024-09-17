#include "lab/spaces/any.h"

namespace lab {

namespace spaces {

AnySpace::AnySpace(const AnySpace& other) : content_(other.content_ ? other.content_->copy() : nullptr) {}

AnySpace& AnySpace::operator=(const AnySpace& other) {
  if (this != &other)
    content_ = other.content_ ? other.content_->copy() : nullptr;
  return *this;
}

AnySpace AnySpace::clone(std::optional<torch::Device> device) const {
  AnySpace clone;
  clone.content_ = content_ ? content_->clone_space(device) : nullptr;
  return clone;
}

torch::Tensor AnySpace::sample(/*torch::Tensor&& mask*/) {
  LAB_CHECK(!is_empty(), "AnySpace is empty");
  return content_->sample(/*std::move(mask)*/);
}

std::shared_ptr<spaces::Space> AnySpace::ptr() const {
  LAB_CHECK(!is_empty(), "AnySpace is empty");
  return content_->ptr();
}

const std::type_info& AnySpace::type_info() const {
  LAB_CHECK(!is_empty(), "AnySpace is empty");
  return content_->type_info;
}

bool AnySpace::is_empty() const noexcept {
  return content_ == nullptr;
}

NamedAnySpace::NamedAnySpace(std::string name, AnySpace any_space)
    : name_(std::move(name)), space_(std::move(any_space)) {}

const std::string& NamedAnySpace::name() const noexcept {
  return name_;
}

AnySpace& NamedAnySpace::space() noexcept {
  return space_;
}

const AnySpace& NamedAnySpace::space() const noexcept {
  return space_;
}

} // namespace spaces

} // namespace lab