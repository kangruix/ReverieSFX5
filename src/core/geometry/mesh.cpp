#pragma once
#include "reverie/core/geometry/mesh.h"
#include <fstream>
#include <sstream>

namespace reverie {
namespace geometry {

void Mesh::write(const std::filesystem::path& objfile) const {
	std::ofstream fout(objfile);

	const point3f* verts = m_vertices.view();
	const vec3i* faces = m_faces.view();

	fout << "# Exported by ReverieSFX" << std::endl;
	for (size_t vi = 0; vi < num_vertices(); ++vi) {
		const point3f& v = verts[vi];
		fout << "v " << std::setprecision(8) <<
			v.x << " " << v.y << " " << v.z << std::endl;
	}
	for (size_t ti = 0; ti < num_faces(); ++ti) {
		const vec3i& f = faces[ti];
		fout << "f " <<
			f.x + 1 << " " << f.y + 1 << " " << f.z + 1 << std::endl;
	}
}

std::string Mesh::to_string() const {
	std::stringstream ss;
	ss << "Mesh[\n" <<
		"  bbox = " << m_bbox.to_string() << ",\n" <<
		"  num_vertices = " << num_vertices() << ",\n" <<
		"  num_faces = " << num_faces() << "\n" <<
		"  device = " << m_device.to_string() << "\n" <<
		"]";
	return ss.str();
}

} // namespace geometry
} // namespace reverie