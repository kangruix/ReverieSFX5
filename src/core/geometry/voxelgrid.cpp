#include "reverie/core/geometry/voxelgrid.h"
#include <sstream>

namespace reverie {
namespace geometry {

VoxelGrid::VoxelGrid(point3i vmin, point3i vmax, vec3f vsize) 
	: dims(vmax.x - vmin.x, vmax.y - vmin.y, vmax.z - vmin.z), vsize(vsize),
	  origin(vmin.x * vsize.x, vmin.y * vsize.y, vmin.z * vsize.z) {
}

std::string VoxelGrid::to_string() const {
	std::stringstream ss;
	ss << "VoxelGrid[\n" <<
		"  dims = [" << std::to_string(dims.x) << ", " << 
			std::to_string(dims.y) << ", " << 
			std::to_string(dims.z) << "]\n" <<
		"  device = " << m_device.to_string() << "\n" <<
		"]";
	return ss.str();
}

} // namespace geometry
} // namespace reverie