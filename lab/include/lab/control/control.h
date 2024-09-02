#pragma once

#include "lab/core.h"
#include "lab/agents/base.h"
#include "lab/envs/base.h"
#include "lab/utils/spec.h"


namespace lab
{
    
namespace control
{

class Session
{
    LAB_ARG(utils::LabSpec, spec);
    LAB_ARG(agents::Agent, agent);
    LAB_ARG(std::shared_ptr<envs::Env>, eval_env);
public:
    Session(const utils::LabSpec& lab_spec);
    void run();
private:
    void run_rl();
    void close();
};

agents::Agent make_agent(const utils::LabSpec& lab_spec);


}

}

