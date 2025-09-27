#pragma once

namespace reverie {

#ifdef __CUDA_ARCH__
#define REV_HOST_DEVICE __host__ __device__
#else
#define REV_HOST_DEVICE
#endif

/// \private
#ifndef __CUDACC__
struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1)
        : x(x_), y(y_), z(z_) {}
};
#endif

} // namespace reverie