#include "reverie/core/geometry/voxelizer.h"
#include "voxelizer.inl"  // Kernel implementations

namespace reverie {
namespace geometry {

template bool Voxelizer::voxelize_surface<DeviceType::CUDA>(const Mesh&);
template bool Voxelizer::voxelize_solid<DeviceType::CUDA>(const Mesh&);

} // namespace geometry
} // namespace reverie