#pragma once

#include "lab/common/common.h"
#include "lab/utils/env.h"
#include "lab/utils/spec.h"

namespace lab {
namespace agents {
class Memory {
 public:
  using ExperienceDict = torch::Dict<std::string, torch::List<torch::IValue>>;

  LAB_ARG(utils::MemorySpec, spec);
  LAB_ARG(utils::StepResult, most_recent);
  LAB_ARG(ExperienceDict, experiences);
  LAB_ARG(int64_t, size) = 0;
  LAB_ARG(int64_t, seen_size) = 0;
  LAB_ARG(bool, ready) = false;
  LAB_ARG(std::vector<std::string>, keys) = {"state", /*"next_state",*/ "action", "reward", "terminated", "truncated"};

 public:
  Memory(const utils::MemorySpec& spec);
  LAB_DEFAULT_CONSTRUCT(Memory);

  virtual void reset();

  virtual void update(const utils::StepResult& result) = 0;

  virtual ExperienceDict sample() = 0;

  torch::List<torch::IValue> get_experiences(const std::string& key) const;

  std::ostream& operator<<(std::ostream& stream);

  void save(torch::serialize::OutputArchive& archive) const;

  void load(torch::serialize::InputArchive& archive);

 private:
  virtual void add_experience(const utils::StepResult& result) = 0;

  void pretty_print(std::ostream& stream, const std::string& indentation) const;

  template <typename Func>
  void pretty_print_list(std::ostream& stream, const std::string& key, const std::string& indentation, Func get_value)
      const;
};

template <typename Func>
void Memory::pretty_print_list(
    std::ostream& stream,
    const std::string& key,
    const std::string& indentation,
    Func get_value) const {
  const std::string next_indentation = indentation + "  ";
  stream << indentation << key << "(\n";
  for (const auto& value : experiences_.at(key))
    stream << next_indentation << get_value(value) << "\n";
  stream << indentation << ")\n";
}

torch::serialize::OutputArchive& operator<<(
    torch::serialize::OutputArchive& archive,
    const std::shared_ptr<Memory>& memory);

torch::serialize::InputArchive& operator>>(
    torch::serialize::InputArchive& archive,
    const std::shared_ptr<Memory>& memory);

} // namespace agents
} // namespace lab