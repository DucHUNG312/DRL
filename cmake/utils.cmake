macro(cuda_get_nvcc_gencode_flag store_var)
  cuda_select_nvcc_arch_flags(${store_var} ${CUDA_ARCH_LIST})
endmacro()

# EXPERIMENT_SPEC_DIR

add_compile_definitions(EXPERIMENT_SPEC_DIR="${CMAKE_SOURCE_DIR}/spec/experiment/")
