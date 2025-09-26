#pragma once
#include "reverie/core/base.h"
#include "reverie/core/geometry/bbox.h"
#include <array>

namespace reverie {
namespace geometry {

class VoxelGrid : public ReverieBase {
public:
	const dim3 dims;		///< grid dimensions (nx, ny, nz)
	const vec3f vsize;		///< voxel size (dx, dy, dz)

	const point3f origin;	///< min corner (world-space)

	VoxelGrid(point3i vmin, point3i vmax, vec3f vsize);
	~VoxelGrid() = default;

	uint32_t num_voxels() const { return dims.x * dims.y * dims.z; }

	const BBox3f& bbox() const { return m_bbox; }

	std::string to_string() const;

private:
	const BBox3f m_bbox;

public:
	struct View { dim3 dims; vec3f vsize; point3f origin; };
	View view() { return { dims, vsize, origin }; }

	/// Converts flattened index \c idx to (i, j, k) indices
	vec3i ijk(uint32_t idx) const {
		int i = idx % dims.x;
		int j = (idx / dims.x) % dims.y;
		int k = (idx / dims.x) / dims.y;
		return { i, j, k };
	}
	/// Converts (x, y, z) position to (i, j, k) indices
	vec3i ijk(point3f P) const {
		int i = (P.x - origin.x) / vsize.x;
		int j = (P.y - origin.y) / vsize.y;
		int k = (P.z - origin.z) / vsize.z;
		return { i, j, k };
	}

	/// Converts (i, j, k) indices to flattened index \c idx
	uint32_t idx(int i, int j, int k) const {
		return dims.x * dims.y * k + dims.x * j + i;
	}
	/// Converts (x, y, z) position to flattened index \c idx
	uint32_t idx(point3f P) const {
		const auto& [i, j, k] = ijk(P);
		return idx(i, j, k);
	}

	/// Converts (i, j, k) indices to (x, y, z) position
	point3f pos(int i, int j, int k) const {
		return origin + vec3f{ vsize.x * (i + 0.5f), 
			vsize.y * (j + 0.5f), vsize.z * (k + 0.5f) };
	}
	/// Converts flattened index \c idx to (x, y, z) position
	point3f pos(uint32_t idx) const {
		const auto& [i, j, k] = ijk(idx);
		return pos(i, j, k);
	}

	/// \returns true if point \c P is within grid bounds
	bool in_bounds(point3f P) const {
		return m_bbox.contains(P);
	}

	/// \returns indices of neighboring voxels
	std::array<uint32_t, 6> neighbors(uint32_t idx) const {
		return { idx - 1, idx + 1, idx - dims.x, idx + dims.y,
			idx - dims.x * dims.y, idx + dims.x * dims.y };
	}
};
    
} // namespace geometry
} // namespace reverie