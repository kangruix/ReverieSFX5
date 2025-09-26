#pragma once
#include <cstdlib>
#include <cstring>
#ifdef REV_ENABLE_CUDA
#include <cuda_runtime.h>
#include <stdexcept>
#endif

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
#ifdef REV_ENABLE_CUDA
    cudaCheck(cudaMalloc(ptr, size));
#else
	*ptr = std::malloc(size);
#endif
}

/// Copies data between host and device
static inline void revMemcpy(void* dst, const void* src, size_t size,
    revMemcpyKind kind) {
#ifdef REV_ENABLE_CUDA
    cudaCheck(cudaMemcpy(dst, src, size, kind));
#else
    std::memcpy(dst, src, size);
#endif
}

/// Initializes or sets memory to \c value
static inline void revMemset(void* ptr, int value, size_t size) {
#ifdef REV_ENABLE_CUDA
    cudaCheck(cudaMemset(ptr, value, size));
#else
	std::memset(ptr, value, size);
#endif
}

/// Frees memory
static inline void revFree(void* ptr) {
#ifdef REV_ENABLE_CUDA
    cudaCheck(cudaFree(ptr));
#else
	std::free(ptr);
#endif
}

} // namespace reverie