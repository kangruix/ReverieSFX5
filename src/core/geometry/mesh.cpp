#pragma once
#include "reverie/core/geometry/mesh.h"
#include <sstream>

namespace reverie {
namespace geometry {

std::string Mesh::to_string() const {
	std::stringstream ss;
	ss << "Mesh[\n" <<
		"  bbox = " << m_bbox.to_string() << ",\n" <<
		"  num_vertices = " << num_vertices() << ",\n" <<
		"  num_faces = " << num_faces() << "\n" <<
		"]\n";
	return ss.str();
}
    
} // namespace geometry
} // namespace reverie