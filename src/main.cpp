#include "reverie/core/geometry/voxelgrid.h"
#include <iostream>

int main() {
	using namespace reverie::geometry;

	VoxelGrid grid({ -32, -32, -32 }, { 32, 32, 32 }, 0.005);
	std::cout << grid.to_string() << std::endl;

    return 0;
}