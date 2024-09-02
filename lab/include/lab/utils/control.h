#pragma once

#include "lab/core.h"
#include "lab/agents/agent.h"
#include "lab/utils/spec.h"

namespace lab
{

namespace utils
{
    
agents::Agent make_agent(const utils::LabSpec& lab_spec);

}

}