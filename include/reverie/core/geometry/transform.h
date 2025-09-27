#pragma once
#include "reverie/core/vector.h"
#include "reverie/core/buffer.h"

namespace reverie {
namespace geometry {

/// \private
template <typename T>
struct quat {
    T w, x, y, z;

    REV_HOST_DEVICE constexpr quat() : w(1), x(0), y(0), z(0) {}
    REV_HOST_DEVICE constexpr quat(T w_, T x_, T y_, T z_) : w(w_), x(x_), y(y_), z(z_) {}

    REV_HOST_DEVICE quat normalized() const {
        T len = sqrt(w * w + x * x + y * y + z * z);
        return { w / len, x / len, y / len, z / len };
    }
    REV_HOST_DEVICE T distance(const quat& o) const { 
        return 1 - (w * o.w + x * o.x + y * o.y + z * o.z);
    }
    quat slerp(T t, quat o) const;
    std::string to_string() const { return "[" + std::to_string(w) + ", " + std::to_string(x) 
        + ", " + std::to_string(y) + ", " + std::to_string(z) + "]"; }
};

template <typename T>
struct Transform {
    vec3<T> translation;
    quat<T> rotation;

    REV_HOST_DEVICE constexpr Transform() = default;
    REV_HOST_DEVICE constexpr Transform(const vec3<T>& t, const quat<T>& q)
        : translation(t), rotation(q) {}

    REV_HOST_DEVICE point3<T> apply(const point3<T>& p) const {
        vec3<T> rp = apply(vec3<T>{p.x, p.y, p.z});
        return point3<T>{rp.x + translation.x, rp.y + translation.y, rp.z + translation.z};
    }
    REV_HOST_DEVICE vec3<T> apply(const vec3<T>& v) const {
        vec3<T> qv = vec3<T>{ rotation.x, rotation.y, rotation.z };
        return v + T(2) * qv.cross(qv.cross(v) + rotation.w * v);
    }
    REV_HOST_DEVICE bool is_identity() const {
        return translation.x == 0 && translation.y == 0 && translation.z == 0 &&
               rotation.w == 1 && rotation.x == 0 && rotation.y == 0 && rotation.z == 0;
	}
    std::string to_string() const;
};

using quatf = quat<float>;
using Transform3f = Transform<float>;

template <DeviceType D>
void apply_transform(Buffer<point3f>& points, const Transform3f& transform);

void apply_transform(Buffer<point3f>& points, const Transform3f& transform);

} // namespace geometry
} // namespace reverie