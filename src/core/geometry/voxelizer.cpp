#include "reverie/core/geometry/voxelizer.h"
#include <fstream>
#ifndef REV_ENABLE_CUDA
#include "voxelizer.inl"  // Kernel implementations
#endif

namespace reverie {
namespace geometry {

Voxelizer::Voxelizer(ref<VoxelGrid> grid)
	: m_grid(grid), 
	  m_bitmask(grid->num_voxels() / 32) {
	m_bitmask.fill(0);
}

void Voxelizer::clear() {
	m_bitmask.fill(0);
}

void Voxelizer::log(const std::filesystem::path& logfile) const {
	std::ofstream fout(logfile, std::ofstream::binary);

	const int* bitmask = m_bitmask.view();
	for (int idx = 0; idx < m_grid->num_voxels(); ++idx) {
		int bit = 31 - (idx % 32);
		int mask = 1 << bit;

		if (bitmask[idx / 32] & mask) {
			point3f P = m_grid->pos(idx);
			fout.write(reinterpret_cast<const char*>(&P), sizeof(point3f));
		}
	}
}

#ifndef REV_ENABLE_CUDA
template bool Voxelizer::voxelize_surface<DeviceType::CPU>(const Mesh&);
template bool Voxelizer::voxelize_solid<DeviceType::CPU>(const Mesh&);
#endif

} // namespace geometry
} // namespace reverie