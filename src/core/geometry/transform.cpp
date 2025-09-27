#include "reverie/core/geometry/transform.h"
#include "reverie/core/backend/kernels.h"
#include <sstream>
#include <limits>

namespace reverie {
namespace geometry {

template <> void apply_transform<DeviceType::CPU>(
    Buffer<point3f>& points, const Transform3f& transform) {
    assert(points.device().type == DeviceType::CPU);

	#pragma omp parallel for
	for (int i = 0; i < points.size(); ++i) {
		points[i] = transform.apply(points[i]);
	}
}

void apply_transform(Buffer<point3f>& points, const Transform3f& transform) {
    if (points.device().type == DeviceType::CPU) {
        apply_transform<DeviceType::CPU>(points, transform);
    }
#ifdef REV_ENABLE_CUDA
    else if (points.device().type == DeviceType::CUDA) {
        apply_transform<DeviceType::CUDA>(points, transform);
    }
#endif
}

template <typename T>
quat<T> quat<T>::slerp(T t, quat o) const {
    T cos_theta = w * o.w + x * o.x + y * o.y + z * o.z;

    if (cos_theta < 0) {
        o.w = -o.w; o.x = -o.x; o.y = -o.y; o.z = -o.z;
        cos_theta = -cos_theta;
    }
    if (cos_theta > 1 - std::numeric_limits<T>::epsilon() * 10) {
        return quat{(1 - t) * w + t * o.w, (1 - t) * x + t * o.x,
            (1 - t) * y + t * o.y, (1 - t) * z + t * o.z}.normalized();
    }
    T theta = acos(fmin(fmax(cos_theta, -1), 1));
    T sin_theta = sin(theta);

    T s0 = sin((1 - t) * theta) / sin_theta;
    T s1 = sin(t * theta) / sin_theta;
    return quat{s0 * w + s1 * o.w, s0 * x + s1 * o.x,
        s0 * y + s1 * o.y, s0 * z + s1 * o.z}.normalized();
}

template struct quat<float>;

template <typename T>
std::string Transform<T>::to_string() const {
    std::stringstream ss;
    ss << "Transform[\n" <<
        "  translation = " << translation.to_string() << ",\n" <<
        "  rotation = " << rotation.to_string() << "\n" <<
        "]";
    return ss.str();
}

template struct Transform<float>;

} // namespace geometry
} // namespace reverie