#include "reverie/core/geometry/voxelizer.h"
#include "reverie/core/backend/kernels.h"
#include "reverie/core/backend/atomics.h"
#include <cassert>

namespace reverie {
namespace geometry {

/**
 * 
 */
template <typename T>
REV_REGISTER_KERNEL(voxelize_surface, 
	int* __restrict d_voxels, dim3 dims, vec3<T> vsize, point3f origin,
	const point3f* __restrict d_vertices, const vec3i* __restrict d_faces, int num_triangles,
	int* __restrict d_counter) {
	constexpr T eps = 1e-6;

#ifdef __CUDACC__
	int thread = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = gridDim.x * blockDim.x;
#else
	int stride = gridDim.x * blockDim.x;
	#pragma omp parallel for
	for (int thread = 0; thread < gridDim.x * blockDim.x; ++thread) {
#endif
	for (int ti = thread; ti < num_triangles; ti += stride) {

		// COMPUTE COMMON TRIANGLE PROPERTIES
		vec3i face = d_faces[ti];
		vec3<T> v0 = static_cast<vec3<T>>(d_vertices[face.x] - origin);
		vec3<T> v1 = static_cast<vec3<T>>(d_vertices[face.y] - origin);
		vec3<T> v2 = static_cast<vec3<T>>(d_vertices[face.z] - origin);

		// Edge vectors
		vec3<T> e0 = v1 - v0;
		vec3<T> e1 = v2 - v1;
		vec3<T> e2 = v0 - v2;

		// Normal vector pointing up from the triangle
		vec3<T> n = e0.cross(e1).normalized();

		// COMPUTE TRIANGLE BBOX IN GRID
		vec3i min = static_cast<vec3i>(v0.min(v1).min(v2) / vsize.x);
		vec3i max = static_cast<vec3i>(v0.max(v1).max(v2) / vsize.y);
		min = min.max(vec3i(0, 0, 0));
		max = max.min(vec3i(dims.x - 1, dims.y - 1, dims.z - 1));

		// PREPARE PLANE TEST PROPERTIES
		vec3<T> c(0, 0, 0);
		if (n.x > 0) { c.x = vsize.x; }
		if (n.y > 0) { c.y = vsize.y; }
		if (n.z > 0) { c.z = vsize.z; }
		T d1 = n.dot(c - v0);
		T d2 = n.dot((vsize - c) - v0);

		// PREPARE PROJECTION TEST PROPERTIES
		// XY plane
		vec2<T> n_xy_e0(-(e0.y), e0.x);
		vec2<T> n_xy_e1(-(e1.y), e1.x);
		vec2<T> n_xy_e2(-(e2.y), e2.x);
		if (n.z < 0) {
			n_xy_e0 = -n_xy_e0;
			n_xy_e1 = -n_xy_e1;
			n_xy_e2 = -n_xy_e2;
		}
		T d_xy_e0 = -n_xy_e0.dot(vec2<T>(v0.x, v0.y)) + fmaxf(0, vsize.x * n_xy_e0.x) + fmaxf(0, vsize.y * n_xy_e0.y);
		T d_xy_e1 = -n_xy_e1.dot(vec2<T>(v1.x, v1.y)) + fmaxf(0, vsize.x * n_xy_e1.x) + fmaxf(0, vsize.y * n_xy_e1.y);
		T d_xy_e2 = -n_xy_e2.dot(vec2<T>(v2.x, v2.y)) + fmaxf(0, vsize.x * n_xy_e2.x) + fmaxf(0, vsize.y * n_xy_e2.y);

		// YZ plane
		vec2<T> n_yz_e0(-(e0.z), e0.y);
		vec2<T> n_yz_e1(-(e1.z), e1.y);
		vec2<T> n_yz_e2(-(e2.z), e2.y);
		if (n.x < 0) {
			n_yz_e0 = -n_yz_e0;
			n_yz_e1 = -n_yz_e1;
			n_yz_e2 = -n_yz_e2;
		}
		T d_yz_e0 = -n_yz_e0.dot(vec2<T>(v0.y, v0.z)) + fmaxf(0, vsize.y * n_yz_e0.x) + fmaxf(0, vsize.z * n_yz_e0.y);
		T d_yz_e1 = -n_yz_e1.dot(vec2<T>(v1.y, v1.z)) + fmaxf(0, vsize.y * n_yz_e1.x) + fmaxf(0, vsize.z * n_yz_e1.y);
		T d_yz_e2 = -n_yz_e2.dot(vec2<T>(v2.y, v2.z)) + fmaxf(0, vsize.y * n_yz_e2.x) + fmaxf(0, vsize.z * n_yz_e2.y);

		// XZ plane
		vec2<T> n_xz_e0(-(e0.x), e0.z);
		vec2<T> n_xz_e1(-(e1.x), e1.z);
		vec2<T> n_xz_e2(-(e2.x), e2.z);
		if (n.y < 0) {
			n_xz_e0 = -n_xz_e0;
			n_xz_e1 = -n_xz_e1;
			n_xz_e2 = -n_xz_e2;
		}
		T d_xz_e0 = -n_xz_e0.dot(vec2<T>(v0.z, v0.x)) + fmaxf(0, vsize.z * n_xz_e0.x) + fmaxf(0, vsize.x * n_xz_e0.y);
		T d_xz_e1 = -n_xz_e1.dot(vec2<T>(v1.z, v1.x)) + fmaxf(0, vsize.z * n_xz_e1.x) + fmaxf(0, vsize.x * n_xz_e1.y);
		T d_xz_e2 = -n_xz_e2.dot(vec2<T>(v2.z, v2.x)) + fmaxf(0, vsize.z * n_xz_e2.x) + fmaxf(0, vsize.x * n_xz_e2.y);

		for (int k = min.z; k <= max.z; ++k) {
			for (int j = min.y; j <= max.y; ++j) {
				for (int i = min.x; i <= max.x; ++i) {

					// TRIANGLE PLANE THROUGH BOX TEST
					vec3<T> p(i * vsize.x, j * vsize.y, k * vsize.z);
					T n_dot_p = n.dot(p);
					if ((n_dot_p + d1) * (n_dot_p + d2) > eps) continue;

					// PROJECTION TESTS
					// XY
					vec2<T> p_xy(p.x, p.y);
					if (n_xy_e0.dot(p_xy) + d_xy_e0 < -eps) continue;
					if (n_xy_e1.dot(p_xy) + d_xy_e1 < -eps) continue;
					if (n_xy_e2.dot(p_xy) + d_xy_e2 < -eps) continue;

					// YZ
					vec2<T> p_yz(p.y, p.z);
					if (n_yz_e0.dot(p_yz) + d_yz_e0 < -eps) continue;
					if (n_yz_e1.dot(p_yz) + d_yz_e1 < -eps) continue;
					if (n_yz_e2.dot(p_yz) + d_yz_e2 < -eps) continue;

					// XZ
					vec2<T> p_xz(p.z, p.x);
					if (n_xz_e0.dot(p_xz) + d_xz_e0 < -eps) continue;
					if (n_xz_e1.dot(p_xz) + d_xz_e1 < -eps) continue;
					if (n_xz_e2.dot(p_xz) + d_xz_e2 < -eps) continue;

					uint32_t idx = dims.x * dims.y * k + dims.x * j + i;

					// Bitwise OR
					int bit = 31 - (idx % 32);
					int mask = 1 << bit;
					if (!(revAtomicOr(&d_voxels[idx / 32], mask) & mask)) {
						int count = revAtomicAdd(d_counter, 1);
						//d_metadata[count] = { idx, ti };
					}
				} // x
			} // y
		} // z
	} // ti
#ifndef __CUDACC__
}
#endif
}

/**
 * \brief
 */
template <typename T>
REV_REGISTER_KERNEL(voxelize_solid,
	int* __restrict d_voxels, dim3 dims, vec3<T> vsize, point3f origin,
	const point3f* __restrict d_vertices, const vec3i* __restrict d_faces, int num_triangles,
	int max_i) {
	constexpr T eps = 1e-12;

#ifdef __CUDACC__
	int thread = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = gridDim.x * blockDim.x;
#else
	int stride = gridDim.x * blockDim.x;
	#pragma omp parallel for
	for (int thread = 0; thread < gridDim.x * blockDim.x; ++thread) {
#endif
	for (int ti = thread; ti < num_triangles; ti += stride) {

		// COMPUTE COMMON TRIANGLE PROPERTIES
		vec3i face = d_faces[ti];
		vec3<T> v0 = static_cast<vec3<T>>(d_vertices[face.x] - origin);
		vec3<T> v1 = static_cast<vec3<T>>(d_vertices[face.y] - origin);
		vec3<T> v2 = static_cast<vec3<T>>(d_vertices[face.z] - origin);

		// Enforce counter-clockwise in YZ plane
		bool CCW = (v1.y - v0.y) * (v2.z - v0.z)
			- (v1.z - v0.z) * (v2.y - v0.y) > 0;
		if (!CCW) { vec3<T> tmp = v1; v1 = v2; v2 = tmp; }  // swap

		// Edge vectors
		vec3<T> e0 = v1 - v0;
		vec3<T> e1 = v2 - v1;
		vec3<T> e2 = v0 - v2;

		// Normal vector pointing up from the triangle
		vec3<T> n = e0.cross(e1).normalized();
		if (fabs(n.x) < eps) continue;

		// Project vertices onto YZ plane
		vec2<T> v0_yz(v0.y, v0.z);
		vec2<T> v1_yz(v1.y, v1.z);
		vec2<T> v2_yz(v2.y, v2.z);

		// PREPARE PROJECTION TEST PROPERTIES
		// YZ Plane
		vec2<T> n_yz_e0(-e0.z, e0.y);
		vec2<T> n_yz_e1(-e1.z, e1.y);
		vec2<T> n_yz_e2(-e2.z, e2.y);
		if (n.x < 0) {
			n_yz_e0 = -n_yz_e0;
			n_yz_e1 = -n_yz_e1;
			n_yz_e2 = -n_yz_e2;
		}
		T d_yz_e0 = -n_yz_e0.dot(vec2<T>(v0.y, v0.z));
		T d_yz_e1 = -n_yz_e1.dot(vec2<T>(v1.y, v1.z));
		T d_yz_e2 = -n_yz_e2.dot(vec2<T>(v2.y, v2.z));

		bool f_yz_e0 = (n_yz_e0.x > 0 || (n_yz_e0.x == 0 && n_yz_e0.y < 0));
		bool f_yz_e1 = (n_yz_e1.x > 0 || (n_yz_e1.x == 0 && n_yz_e1.y < 0));
		bool f_yz_e2 = (n_yz_e2.x > 0 || (n_yz_e2.x == 0 && n_yz_e2.y < 0));

		vec2<T> min_yz = v0_yz.min(v1_yz).min(v2_yz);
		vec2<T> max_yz = v0_yz.max(v1_yz).max(v2_yz);
		min_yz = vec2<T>(ceil(min_yz.x / vsize.y - 0.5), ceil(min_yz.y / vsize.z - 0.5));
		max_yz = vec2<T>(floor(max_yz.x / vsize.y - 0.5), floor(max_yz.y / vsize.z - 0.5));

		for (int k = min_yz.y; k <= max_yz.y; ++k) {
			for (int j = min_yz.x; j <= max_yz.x; ++j) {
				vec2<T> p_yz((j + 0.5) * vsize.y, (k + 0.5) * vsize.z);

				if (f_yz_e0 && n_yz_e0.dot(p_yz) + d_yz_e0 < 0) continue;
				else if (!f_yz_e0 && n_yz_e0.dot(p_yz) + d_yz_e0 <= 0) continue;

				if (f_yz_e1 && n_yz_e1.dot(p_yz) + d_yz_e1 < 0) continue;
				else if (!f_yz_e1 && n_yz_e1.dot(p_yz) + d_yz_e1 <= 0) continue;

				if (f_yz_e2 && n_yz_e2.dot(p_yz) + d_yz_e2 < 0) continue;
				else if (!f_yz_e2 && n_yz_e2.dot(p_yz) + d_yz_e2 <= 0) continue;

				// Compute intersection point
				T n_dot_v0 = n.x * v0.x + n.y * v0.y + n.z * v0.z;
				T q = (n_dot_v0 - n.y * p_yz.x - n.z * p_yz.y) / n.x / vsize.x;
				int qbar = floor(q + 0.5);

				for (int i = qbar; i <= max_i; ++i) {
					uint32_t idx = dims.x * dims.y * k + dims.x * j + i;

					// Bitwise XOR
					int bit = 31 - (idx % 32);
					int mask = 1 << bit;
					revAtomicXor(&d_voxels[idx / 32], mask);
				} // x
			} // y
		} // z
	} // ti
#ifndef __CUDACC__
}
#endif
}

template <DeviceType D>
bool Voxelizer::voxelize_surface(const Mesh& mesh) {
	assert(mesh.device().type == D);

	/*int* d_counter = nullptr;
	revMalloc((void**) &d_counter, sizeof(int));
	revMemset(d_counter, 0, sizeof(int));

	int blockDim = 256;  // \todo
	int gridDim = ((mesh.num_faces() + blockDim - 1) / blockDim);

	REV_LAUNCH_KERNEL(voxelize_surface, gridDim, blockDim,
		d_bitmask, m_grid->dims, m_grid->vsize, m_grid->origin,
		mesh.vertices(), mesh.faces(), mesh.num_faces(),
		d_counter);

	int count = 0;
	revMemcpy(&count, d_counter, sizeof(int), DeviceToHost);
	revFree(d_counter);*/

	return true;
}

template <DeviceType D>
bool Voxelizer::voxelize_solid(const Mesh& mesh) {
	assert(mesh.device().type == D);

	/*int blockDim = 256; // \todo
	int gridDim = ((mesh.num_faces() + blockDim - 1) / blockDim);

	REV_LAUNCH_KERNEL(voxelize_solid, gridDim, blockDim,
		d_bitmask, m_grid->dims, static_cast<vec3d>(m_grid->vsize), m_grid->origin,
		mesh.vertices(), mesh.faces(), mesh.num_faces(),
		std::ceil((mesh.bbox().max.x - m_grid->origin.x) / m_grid->vsize.x));*/

	return true;
}

} // namespace geometry
} // namespace reverie