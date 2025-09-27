#pragma once
#include "reverie/core/vector.h"
#include "reverie/core/buffer.h"

namespace reverie {
namespace geometry {

template <typename T>
struct BBox {
    point3<T> min;
    point3<T> max;

    REV_HOST_DEVICE constexpr BBox();
    REV_HOST_DEVICE constexpr BBox(const point3<T>& min_, const point3<T>& max_)
        : min(min_), max(max_) {}

    REV_HOST_DEVICE bool is_valid() const {
        return min.x <= max.x && min.y <= max.y && min.z <= max.z;
	}
    REV_HOST_DEVICE void extend(const point3<T>& p) {
		min = min.min(p); max = max.max(p);
    }
    REV_HOST_DEVICE void extend(const BBox& b) {
		min = min.min(b.min); max = max.max(b.max);
    }
    REV_HOST_DEVICE bool contains(const point3<T>& p) const {
        return p.x >= min.x && p.x <= max.x &&
               p.y >= min.y && p.y <= max.y &&
               p.z >= min.z && p.z <= max.z;
    }
    REV_HOST_DEVICE point3<T> center() const {
        return (min.x + max.x) / 2;
    }
    REV_HOST_DEVICE T distance(const point3<T>& p) const { 
		T dx = (p.x < min.x) ? (min.x - p.x) : 
               (p.x > max.x) ? (p.x - max.x) : 0;
		T dy = (p.y < min.y) ? (min.y - p.y) : 
               (p.y > max.y) ? (p.y - max.y) : 0;
		T dz = (p.z < min.z) ? (min.z - p.z) : 
               (p.z > max.z) ? (p.z - max.z) : 0;
        return sqrt(dx * dx + dy * dy + dz * dz);
    }
    REV_HOST_DEVICE T distance(const BBox& b) const {
		T dx = (b.max.x < min.x) ? (min.x - b.max.x) : 
               (b.min.x > max.x) ? (b.min.x - max.x) : 0;
		T dy = (b.max.y < min.y) ? (min.y - b.max.y) : 
               (b.min.y > max.y) ? (b.min.y - max.y) : 0;
		T dz = (b.max.z < min.z) ? (min.z - b.max.z) : 
               (b.min.z > max.z) ? (b.min.z - max.z) : 0;
        return sqrt(dx * dx + dy * dy + dz * dz);
    }
    std::string to_string() const;
};

using BBox3f = BBox<float>;

template <DeviceType D>
BBox3f compute_bbox(const Buffer<point3f>& points);

BBox3f compute_bbox(const Buffer<point3f>& points);

template<> REV_HOST_DEVICE constexpr BBox<float>::BBox()
    : min(3.402823466e+38F), max(-3.402823466e+38F) {}

} // namespace geometry
} // namespace reverie