#pragma once

#include "lab/common/common.h"
#include "lab/agents/agent.h"
#include "lab/utils/spec.h"

namespace lab
{
    
namespace control
{

class Session
{
    LAB_ARG(utils::LabSpec, spec);
    LAB_ARG(agents::Agent, agent);
    LAB_ARG(double, max_total_reward) = 0;
public:
    Session(const utils::LabSpec& lab_spec);
    void run();
private:
    void run_rl();
    void close();
    void update_total_reward();
};


}

}

