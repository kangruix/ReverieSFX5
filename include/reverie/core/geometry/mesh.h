#pragma once
#include "reverie/core/base.h"
#include "reverie/core/geometry/bbox.h"
#include <filesystem>

namespace reverie {
namespace geometry {

class Mesh : public ReverieBase {
public:
	/// Default constructor: initialize empty Mesh
	Mesh(const Device& device = Device()) : ReverieBase(device) {}

	/// Read Mesh from .obj file
	bool read(const std::filesystem::path& objfile);

	/// Write Mesh to .obj file
	void write(const std::filesystem::path& objfile) const;

	const point3f* vertices() const { return m_vertices.data(); }
	const vec3i* faces() const { return m_faces.data(); }

	size_t num_vertices() const { return m_vertices.size(); }
	size_t num_faces() const { return m_faces.size(); }

	BBox3f bbox() const { return m_bbox; }

	std::string to_string() const;

private:
	Buffer<point3f> m_vertices;
	Buffer<vec3i> m_faces;

	BBox3f m_bbox;

	static constexpr int MAX_FACES = std::numeric_limits<int>::max();
};

} // namespace geometry
} // namespace reverie