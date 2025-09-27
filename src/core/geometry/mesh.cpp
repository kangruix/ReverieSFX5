#include "reverie/core/geometry/mesh.h"
#include <fstream>
#include <sstream>

namespace reverie {
namespace geometry {

bool Mesh::read(const std::filesystem::path& objfile) {
	std::ifstream fin(objfile);
	if (!fin) throw std::runtime_error("Mesh: failed to open file " + objfile.string());

	std::vector<point3f> vertices;
	std::vector<vec3i> faces;

	std::string line, prefix;
	while (std::getline(fin, line)) {
		std::istringstream sstream(line);
		sstream >> prefix;

		if (prefix == "v") {
			point3f v;
			sstream >> v.x >> v.y >> v.z;
			vertices.push_back(v);
		} 
		else if (prefix == "f") {
			std::string f1, f2, f3;
			sstream >> f1 >> f2 >> f3;
			
			auto split = [](const std::string& f) -> int {
				return std::stoi(f.substr(0, f.find('/')));
			};
			faces.push_back({ split(f1) - 1, split(f2) - 1, split(f3) - 1 });
		}
	}
	if (vertices.size() == 0 || faces.size() == 0) return false;

	m_vertices = Buffer<point3f>(vertices, m_device);
	m_faces = Buffer<vec3i>(faces, m_device);
	m_bbox = compute_bbox(m_vertices);

	return true;
}

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