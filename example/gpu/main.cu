#include <lab/lab.h>

template <typename Fn>
LAB_GLOBAL void job_entry(lab::cuda::JobQueue* _job_queue, void* data)
{
    lab::cuda::Context ctx { .job_queue = *_job_queue };

    auto fn_ptr = (Fn *)data;
    (*fn_ptr)(ctx);
    fn_ptr->~Fn();

    atomicSub(&ctx.job_queue.num_outstanding_jobs, 1u);
}

template <typename Fn>
LAB_GLOBAL void set_initial_job_kernel_address(lab::cuda::JobQueue* job_queue)
{
    job_queue->jobs[0].fn = job_entry<Fn>;
}

template <typename Fn>
lab::cuda::JobQueue* init_job_queue(cudaStream_t strm, Fn &&fn)
{
    lab::cuda::JobQueue* job_queue = (lab::cuda::JobQueue*)lab::cuda::alloc_gpu(sizeof(lab::cuda::JobQueue));
    lab::cuda::JobQueue* queue_staging = (lab::cuda::JobQueue*)lab::cuda::alloc_staging(sizeof(lab::cuda::JobQueue));

    queue_staging->job_head = 0;
    queue_staging->num_waiting_jobs = 1;
    queue_staging->num_outstanding_jobs = 0;

    set_initial_job_kernel_address<Fn><<<1, 1, 0, strm>>>(queue_staging);

    queue_staging->jobs[0].arg = &job_queue->job_data.buffer;

    new (&(queue_staging->job_data.buffer)[0]) Fn(std::forward<Fn>(fn));

    lab::cuda::copy_cpu_to_gpu(strm, job_queue, queue_staging, sizeof(lab::cuda::JobQueue));
    LAB_CHECK_CUDA(cudaStreamSynchronize(strm));

    lab::cuda::dealloc_cpu(queue_staging);

    return job_queue;
}

LAB_GLOBAL void job_system(lab::cuda::JobQueue* job_queue)
{
    uint32_t thread_pos = threadIdx.x;

    while (true) 
    {
        uint32_t cur_num_jobs = job_queue->num_waiting_jobs;

        if (thread_pos < cur_num_jobs) 
        {
            lab::cuda::Job job = job_queue->jobs[job_queue->job_head + thread_pos];
            job.fn<<<1, 1>>>(job_queue, job.arg);
        }

        __syncthreads();

        if (thread_pos == 0)
        {
            atomicSub(&job_queue->num_waiting_jobs, cur_num_jobs);
            atomicAdd(&job_queue->num_outstanding_jobs, cur_num_jobs);
            atomicAdd(&job_queue->job_head, cur_num_jobs);
        }

        if (job_queue->num_outstanding_jobs == 0) 
        {
            break;
        }
    }
}

void init_training()
{
    auto strm = lab::cuda::make_stream();

    int v = 5;
    lab::cuda::JobQueue *job_queue = init_job_queue(strm, [v] __device__ (lab::cuda::Context& ctx) 
    {
        printf("Hi %d\n", v);
    });

    job_system<<<1, 1024, 0, strm>>>(job_queue);

    LAB_CHECK_CUDA(cudaStreamSynchronize(strm));
}

int main(int argc, char** argv)
{
    LAB_INIT_LOG();

    init_training();

    LAB_SHUTDOWN_LOG();

    return 0;
}