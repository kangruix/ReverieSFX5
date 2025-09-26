#include "reverie/core/geometry/bbox.h"
#include "reverie/core/backend/kernels.h"

namespace reverie {
namespace geometry {

__global__ void compute_bbox_cu(
    const point3f* __restrict d_points, int num_points, BBox3f* d_bbox) {

    extern __shared__ BBox3f s_bbox[];
    int thread = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    BBox3f l_bbox;  // local BBox
    for (int i = thread; i < num_points; i += stride) {
        l_bbox.extend(d_points[i]);
    }
    s_bbox[thread] = l_bbox;
    __syncthreads();

    for (int o = blockDim.x >> 1; o > 0; o >>= 1) {
        if (thread < o) { s_bbox[thread].extend(s_bbox[thread + o]); }
        __syncthreads();
    }
    if (thread == 0) { d_bbox[0] = s_bbox[0]; }
}

template<> BBox3f compute_bbox<DeviceType::CUDA>(
    const point3f* d_points, int num_points) {
    int blockDim = 256;
    int gridDim = 1;
	int sharedMem = blockDim * sizeof(BBox3f);

    BBox3f* d_bbox;
    revMalloc((void**) &d_bbox, sizeof(BBox3f));

    compute_bbox_cu<<<gridDim, blockDim, sharedMem>>>(
        d_points, num_points, d_bbox);

    BBox3f h_bbox;
    revMemcpy(&h_bbox, d_bbox, sizeof(BBox3f), DeviceToHost);
    revFree(d_bbox);

    return h_bbox;
}

} // namespace geometry
} // namespace reverie