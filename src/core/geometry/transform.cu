#include "reverie/core/geometry/transform.h"
#include "reverie/core/backend/kernels.h"

namespace reverie {
namespace geometry {

__global__ void apply_transform_cu(
	point3f* __restrict d_points, Transform3f transform, int num_points) {
	int thread = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = gridDim.x * blockDim.x;
	
	for (int i = thread; i < num_points; i += stride)
		d_points[i] = transform.apply(d_points[i]);
}

template<> void apply_transform<GPU>(
	Buffer<point3f>& points, const Transform3f& transform) {
	int blockDim = 256;
	int gridDim = ((points.size() + blockDim - 1) / blockDim);

	apply_transform_cu<<<gridDim, blockDim>>>(
		points.data(), transform, points.size());
}

} // namespace geometry
} // namespace reverie