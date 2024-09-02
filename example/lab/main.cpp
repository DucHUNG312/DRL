#include <lab/lab.h>

using namespace lab;

int main(int argc, char** argv)
{
    LAB_INIT_LOG();

    std::string path = utils::join_paths(std::string(EXPERIMENT_SPEC_DIR), "reinforce/reinforce_cartpole.json");
    utils::SpecLoader::load_specs_from_file(path, "reinforce_cartpole");
    
    control::Session session(utils::SpecLoader::specs);

    session.run();

    LAB_SHUTDOWN_LOG();

    return 0;
}