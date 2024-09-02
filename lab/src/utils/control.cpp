#include "lab/utils/control.h"
#include "lab/agents/body.h"
#include "lab/envs/base.h"

namespace lab
{

namespace utils
{

agents::Agent make_agent(const utils::LabSpec& lab_spec)
{
    std::shared_ptr<envs::Env> env = utils::create_env(lab_spec.env);
    std::shared_ptr<agents::Body> body = std::make_shared<agents::Body>(env, lab_spec.body);
    return agents::Agent(body, lab_spec.agent);
}


}

}