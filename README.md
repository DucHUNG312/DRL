## Build

Project will build with GPU support by default, you can set `LAB_GPU` to `OFF` to run on CPU only.

```
mkdir build
cd build
cmake .. -DLAB_GPU=OFF
make -j 8
```

## GPU Support

Check your GPU's compute capabiliy version [here](https://developer.nvidia.com/cuda-gpus) and set it to `CMAKE_CUDA_ARCHITECTURE`. Otherwise project will choose default the compute capabiliy is 5.2

```
cmake .. -DCMAKE_CUDA_ARCHITECTURE=${GPU_COMPUTE_CAPABILITY}
```