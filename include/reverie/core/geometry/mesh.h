#pragma once
#include "reverie/core/base.h"
#include "reverie/core/geometry/bbox.h"

namespace reverie {
namespace geometry {

class Mesh : public ReverieBase {
public:
	Mesh() = default;

	size_t num_vertices() const { return m_vertices.size(); }
	size_t num_faces() const { return m_faces.size(); }

	BBox3f bbox() const { return m_bbox; }

	std::string to_string() const;

private:
	Buffer<point3f> m_vertices;
	Buffer<vec3i> m_faces;
	Device m_device;

	BBox3f m_bbox;

public:
	struct View {
		const point3f* d_vertices; const vec3i* d_faces;
		int num_vertices; int num_faces;
	};

	View view() const {
		return { m_vertices.get(), m_faces.get(),
			(int) m_vertices.size(), (int) m_faces.size() };
	}
};

} // namespace geometry
} // namespace reverie