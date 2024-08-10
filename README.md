## Build

Check your GPU's compute capabiliy version [here](https://developer.nvidia.com/cuda-gpus).

```
mkdir build && cd build
cmake .. -CMAKE_CUDA_ARCHITECTURE=${GPU_COMPUTE_CAPABILITY}
make -j 8
```