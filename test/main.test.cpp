#include "main.h"

using namespace lab;

int main(int argc, char** argv)
{
    Logger::init();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}