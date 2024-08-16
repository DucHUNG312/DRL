## Build

The project will build with GPU support by default. You can set `LAB_GPU` to `OFF` to run it on the CPU only.

```
mkdir build
cd build
cmake .. -DLAB_GPU=OFF
make -j 8
```

## GPU Support

Check your GPU's compute capability version [here](https://developer.nvidia.com/cuda-gpus) and set it using `CMAKE_CUDA_ARCHITECTURE`. Otherwise, the default compute capability of 5.2 will be chosen.

```
cmake .. -DCMAKE_CUDA_ARCHITECTURE=${GPU_COMPUTE_CAPABILITY}
```