#include "lab/agents/base.h"

namespace lab
{

namespace agents
{

Agent::Agent(const std::shared_ptr<Body>& body, const utils::AgentSpec& spec)
    : body_(body), spec_(std::move(spec))
{
}

}

}