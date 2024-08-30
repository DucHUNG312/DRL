#include <lab/lab.h>

int main(int argc, char** argv)
{
    LAB_INIT_LOG();

    lab::envs::CartPole env;
    // // env.enable_rendering();

    while(!env.done()) 
    {
        env.step(env.sample());
        LAB_LOG_DEBUG(env.get_result());
    }

    env.close();
    LAB_SHUTDOWN_LOG();

    return 0;
}