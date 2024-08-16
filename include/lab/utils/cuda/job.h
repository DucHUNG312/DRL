#pragma once

#include "lab/core.h"
#include "lab/utils/cuda/cuda_utils.h"

namespace lab
{

namespace cuda
{

struct JobQueue;

struct Job 
{
    using FuncPtr = void (*)(JobQueue* job_queue, void*);
    FuncPtr fn;
    void* arg;
};

struct alignas(64) JobData 
{
    char buffer[10 * 1024 * 1024];
};

struct JobQueue 
{
    Job jobs[16384];
    uint32_t job_head;
    uint32_t num_waiting_jobs;
    uint32_t num_outstanding_jobs;
    JobData job_data;
};

class Context 
{
public:
    JobQueue& job_queue;
};

}

}