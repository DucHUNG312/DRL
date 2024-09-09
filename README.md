## Requirements

- Ubuntu 22.04

- CUDA Toolkit >= 12.5: [link](https://developer.nvidia.com/cuda-downloads).

- Libtorch 2.4.0: [GPU version](https://download.pytorch.org/libtorch/cu124/) or [CPU version](https://download.pytorch.org/libtorch/cpu/).

- CMake >= 3.25

- Clang >= 17.0.6 or GCC >= 12.3.0

## Build

- Note: Ensure that the CUDA Toolkit is correctly installed and properly configured. Also export the location of LibTorch, so CMake can locate it during the build process. (```export Torch_DIR=PATH_TO_LIBTORCH```).

```
mkdir build
cd build
cmake -G Ninja ..
ninja -j8
```

## TODO

- [x] Create a trainning pipeline.
- [ ] Add more algorithms.
- [ ] Support more environments.
- [ ] Support real time rendering.