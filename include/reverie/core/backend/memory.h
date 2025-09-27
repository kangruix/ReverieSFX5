#pragma once
#include <cstdlib>
#include <cstring>
#ifdef REV_ENABLE_CUDA
#include <cuda_runtime.h>
#include <stdexcept>
#endif

/**
 * Low-level memory management API
 * 
 * The behaviors of revMalloc, revMemcpy, revMemset, and revFree depend
 * on the translation unit they are invoked from:
 * 
 * If invoked from a .cu file (i.e., compiled with nvcc), these functions
 * manage memory on the GPU using CUDA APIs.
 * 
 * If invoked from a regular .cpp file, these functions revert to
 * standard C/C++ CPU memory management (malloc, memcpy, etc.).
 * 
 * Along with the other backend components, this allows for writing 
 * device-agnostic code that compiles to both valid C++ and CUDA kernels.
 */

namespace reverie {

/// CUDA error checking helper
#ifdef REV_ENABLE_CUDA
static inline void cudaCheck(cudaError_t code) {
    if (code != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(code));
}
#endif

/// Memcpy directions
#ifdef REV_ENABLE_CUDA
using revMemcpyKind = cudaMemcpyKind;
constexpr revMemcpyKind HostToDevice = cudaMemcpyHostToDevice;
constexpr revMemcpyKind DeviceToHost = cudaMemcpyDeviceToHost;
constexpr revMemcpyKind DeviceToDevice = cudaMemcpyDeviceToDevice;
#else
enum revMemcpyKind { HostToDevice, DeviceToHost, DeviceToDevice };
#endif

/// Allocates memory on device
static inline void revMalloc(void** ptr, size_t size) {
#ifdef __CUDACC__
    cudaCheck(cudaMalloc(ptr, size));
#else
	*ptr = std::malloc(size);
#endif
}

/// Copies data between host and device
static inline void revMemcpy(void* dst, const void* src, size_t size,
    revMemcpyKind kind) {
#ifdef __CUDACC__
    cudaCheck(cudaMemcpy(dst, src, size, kind));
#else
    std::memcpy(dst, src, size);
#endif
}

/// Initializes or sets memory to \c value
static inline void revMemset(void* ptr, int value, size_t size) {
#ifdef __CUDACC__
    cudaCheck(cudaMemset(ptr, value, size));
#else
	std::memset(ptr, value, size);
#endif
}

/// Frees memory
static inline void revFree(void* ptr) {
#ifdef __CUDACC__
    cudaCheck(cudaFree(ptr));
#else
	std::free(ptr);
#endif
}

} // namespace reverie