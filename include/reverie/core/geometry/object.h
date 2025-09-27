#pragma once
#include "reverie/core/geometry/mesh.h"
#include "reverie/core/geometry/transform.h"

namespace reverie {
namespace geometry {

// Forward declarations
class Animation;

class Object : public ReverieBase {
public:
    /// Constructor: initialize from Properties
    Object(const Properties& props);

    /// Updates Object animation at a given time
    bool update_animation(double time);

    /// \returns true if this Object is also a Source
    //bool is_source() const { return m_source; }

    const uint8_t id;   ///< unique id (1-indexed)

    const std::string& name() const { return m_name; }

    const Mesh& mesh() const { return m_mesh; }
    const BBox3f& bbox() const { return m_mesh.bbox(); }
	const Transform3f& transform() const { return m_transform; }

    std::string to_string() const override;

private:
    std::string m_name;

    Mesh m_mesh;
	Transform3f m_transform;

    ref<Animation> m_animation;
    //ref<signal::Source> m_source;

    static uint8_t s_global_id;
};

} // namespace geometry
} // namespace reverie