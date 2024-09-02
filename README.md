## This project has been tested on

- Ubuntu 22.04
- CUDA 12.5
- GCC 12.3.0
- PyTorch 2.4.0

## Build

The project will build with GPU support by default. You can set `LAB_GPU` to `OFF` to run it on the CPU only.

```
mkdir build
cd build
cmake -G Ninja .. -DLAB_GPU=OFF
ninja -j8
```

## GPU Support

Check your GPU's compute capability version [here](https://developer.nvidia.com/cuda-gpus) and set it using `CMAKE_CUDA_ARCHITECTURE`. Otherwise, the default compute capability of 5.2 will be chosen.

```
cmake -G Ninja .. -DCMAKE_CUDA_ARCHITECTURE=${GPU_COMPUTE_CAPABILITY}
```

## TODO

- [ ] Create a trainning framework.
- [ ] Add DRL algorithms.
- [ ] Support more environments.
- [ ] Support real time rendering.