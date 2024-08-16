#include <lab/lab.h>

int main(int argc, char** argv)
{
    LAB_INIT_LOG();

    lab::envs::CartPole env;

    // Not support rendering yet!
    //env.enable_rendering();

    for(int i = 0; i < 50; i ++) 
    {
        env.step(env.action_space().sample());
        LAB_LOG_DEBUG(env.state());
    }

    LAB_SHUTDOWN_LOG();

    return 0;
}