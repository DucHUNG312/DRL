## This project has been tested on

- Ubuntu 22.04
- CUDA 12.5
- GCC 12.3.0
- PyTorch 2.4.0

## Project structure

- `lab/`: Contains the core code for the framework. This directory holds all the essential components, utilities, and main logic required for the framework's functionality.

- `renderer/`: A separate project dedicated to rendering support for the framework. This folder includes modules and tools specifically for rendering tasks.

## Build

- Check your GPU's compute capability version [here](https://developer.nvidia.com/cuda-gpus) and set it using `CMAKE_CUDA_ARCHITECTURE`. Otherwise, the default compute capability of 5.2 will be chosen.

- Note: The project will build with GPU support by default. You can set `LAB_GPU` to `OFF` to build it with only CPU support.

- Note: CMake will automatically clone and download third-party libraries, which may take a few minutes to complete.

```
mkdir build
cd build
cmake -G Ninja .. -DCMAKE_CUDA_ARCHITECTURE=YOUR_GPU_COMPUTE_CAPABILITY
ninja -j8
```

## TODO

- [x] Create a trainning pipeline.
- [ ] Add more algorithms.
- [ ] Support more environments.
- [ ] Support real time rendering.