#pragma once


namespace lab
{
namespace envs
{
class Env;
class CartPole;
}

namespace agents
{
class Algorithm;
class Reinforce;

class Memory;
class OnPolicyReplay;

class NetImpl;
class MLPNetImpl;

class Agent;
}

namespace distributions
{
class Distribution;
}

namespace utils
{
struct LabSpec;
struct AlgorithmSpec;
struct EnvSpec;
struct MemorySpec;
struct LrSchedulerSpec;
struct NetSpec;

struct ActionPolicy;

struct StepResult;
class Clock;
}
}