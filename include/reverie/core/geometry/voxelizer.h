#pragma once
#include "reverie/core/geometry/mesh.h"
#include "reverie/core/geometry/voxelgrid.h"
#include <filesystem>

namespace reverie {
namespace geometry {

class Voxelizer {
public:
	Voxelizer(ref<VoxelGrid> grid);

	template <DeviceType D>
	bool voxelize_surface(const Mesh& mesh);

	template <DeviceType D>
	bool voxelize_solid(const Mesh& mesh);

	/// Clears bitmask
	void clear();

	/// Logs occupied voxel centers to binary file
	void log(const std::filesystem::path& logfile) const;

private:
	ref<VoxelGrid> m_grid = nullptr;

	/// Dense bitmask for voxel occupancy (0 - empty, 1 - occupied)
	Buffer<int> m_bitmask;
};

} // namespace geometry
} // namespace reverie