#pragma once
#include "reverie/core/backend/memory.h"
#ifdef REV_ENABLE_CUDA
#include <device_launch_parameters.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * \file core/backend/kernels.h
 * 
 * REV_REGISTER_KERNEL(my_kernel, ...) {
 *     // kernel code
 * }
 * REV_LAUNCH_KERNEL(my_kernel, gridDim, blockDim, ...);
 * 
 * On CUDA builds:
 * - registers as __global__ void my_kernel_cu(...) 
 * - launches as my_kernel_cu<<<gridDim, blockDim>>>(...)
 * 
 * On CPU builds:
 * - registers as void my_kernel_cpu(dim3 gridDim, dim3 blockDim, ...)
 * - launches as my_kernel_cpu(gridDim, blockDim, ...)
 */

#ifdef __CUDACC__
#define REV_REGISTER_KERNEL(name, ...) \
    __global__ void name##_cu(__VA_ARGS__)

#define REV_LAUNCH_KERNEL(name, gridDim, blockDim, ...) \
    name##_cuda<<<gridDim, blockDim>>>(__VA_ARGS__); \
    cudaCheck(cudaGetLastError())
#else
#define REV_REGISTER_KERNEL(name, ...) \
    void name##_cpu(dim3 gridDim, dim3 blockDim, __VA_ARGS__)

#define REV_LAUNCH_KERNEL(name, gridDim, blockDim, ...) \
    name##_cpu(gridDim, blockDim, __VA_ARGS__)
#endif