#pragma once

namespace lab {
namespace envs {
class Env;
class CartPole;
} // namespace envs

namespace agents {
class Algorithm;
class Reinforce;

class Memory;
class OnPolicyReplay;

class NetImpl;
class MLPNetImpl;

class Agent;
} // namespace agents

namespace distributions {
class Distribution;
}

namespace utils {
struct LabSpec;
struct AlgorithmSpec;
struct EnvSpec;
struct MemorySpec;
struct LrSchedulerSpec;
struct NetSpec;

struct ActionPolicy;

struct StepResult;
class Clock;
} // namespace utils
} // namespace lab