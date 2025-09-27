#pragma once
#ifdef _OPENMP
#include <omp.h>
#endif

namespace reverie {

#ifdef __CUDA_ARCH__
/// CUDA atomic operations
static __device__ int revAtomicOr(int* address, int val) { return atomicOr(address, val); }
static __device__ int revAtomicXor(int* address, int val) { return atomicXor(address, val); }
template <typename T>
static __device__ T revAtomicAdd(T* address, T val) { return atomicAdd(address, val); }
#else
/// OpenMP atomic operations
/// \todo once MSVC supports omp atomic capture, replace critical sections
static inline int revAtomicOr(int* address, int val) {
    int old;
    #pragma omp critical
    {
        old = *address; *address |= val;
    }
    return old;
}
static inline int revAtomicXor(int* address, int val) {
    int old;
    #pragma omp critical
    {
        old = *address; *address ^= val;
    }
    return old;
}
template <typename T>
static inline T revAtomicAdd(T* address, T val) {
    T old;
    #pragma omp critical
    {
        old = *address; *address += val;
    }
    return old;
}
#endif

} // namespace reverie