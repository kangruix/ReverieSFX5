#include "reverie/fdtd/grid.h"
#include "reverie/core/backend/memory.h"

namespace reverie {
namespace fdtd {

Grid::Grid(point3i vmin, point3i vmax, Config conf, 
    int step, Device device) 
    : VoxelGrid(vmin, vmax, conf.cellsize, device),
      m_config(conf), step(step),
	  dt(1. / conf.samplerate), 
      lambda(conf.sound_speed * dt / conf.cellsize),
      m_pressure(num_voxels() * 2, device), 
      m_metadata(num_voxels(), device) {
 
	if (lambda * lambda > 1. / 3) { throw std::runtime_error("CFL limit exceeded!"); }
	size_t gridsize = num_voxels();

    // Set initial values
	m_pressure.fill(0);

	// Set device pointers
	d_p0 = m_pressure.data();
	d_p1 = d_p0 + gridsize;
	d_md = m_metadata.data();
}

Grid::~Grid() {
    d_p0 = d_p1 = nullptr;
    d_md = nullptr;
}

} // namespace fdtd
} // namespace reverie