#pragma once
#include "reverie/core/platform.h"
#define _USE_MATH_DEFINES
#include <cmath>
#include <string>

namespace reverie {

// 2D types

#define REV_IMPORT_ARRAY2(T, array2) \
    T x, y; \
    REV_HOST_DEVICE constexpr array2(T c = 0) : x(c), y(c) {} \
    REV_HOST_DEVICE constexpr array2(T x_, T y_) : x(x_), y(y_) {} \
    \
    REV_HOST_DEVICE array2 operator*(T s) const { return { x * s, y * s }; } \
    REV_HOST_DEVICE array2 operator/(T s) const { return { x / s, y / s }; } \
    REV_HOST_DEVICE array2 operator-() const { return { -x, -y }; } \
    \
    REV_HOST_DEVICE array2 min(const array2& o) const { return { x < o.x ? x : o.x, y < o.y ? y : o.y }; } \
    REV_HOST_DEVICE array2 max(const array2& o) const { return { x > o.x ? x : o.x, y > o.y ? y : o.y }; } \
    \
    template <typename U> REV_HOST_DEVICE explicit constexpr array2(const array2<U>& o) \
        : x(static_cast<T>(o.x)), y(static_cast<T>(o.y)) {} \
    \
    std::string to_string() const { return "[" + std::to_string(x) + ", " + std::to_string(y) + "]"; }

/// \private
template <typename T> struct vec2 {
    REV_IMPORT_ARRAY2(T, vec2)

    REV_HOST_DEVICE vec2 operator+(const vec2& o) const { return { x + o.x, y + o.y }; }
    REV_HOST_DEVICE vec2 operator-(const vec2& o) const { return { x - o.x, y - o.y }; }

    REV_HOST_DEVICE T dot(const vec2& o) const { return x * o.x + y * o.y; }

    REV_HOST_DEVICE T norm() const { return sqrt(x * x + y * y); }
    REV_HOST_DEVICE vec2 normalized() const { T len = norm(); return len != 0 ? *this / len : *this; }
};
template<typename T> REV_HOST_DEVICE vec2<T> operator*(T s, const vec2<T>& v) { return { s * v.x, s * v.y }; }

/// \private
template <typename T> struct point2 {
    REV_IMPORT_ARRAY2(T, point2)

    REV_HOST_DEVICE point2  operator+(const vec2<T>& o) const { return { x + o.x, y + o.y }; }
    REV_HOST_DEVICE point2  operator-(const vec2<T>& o) const { return { x - o.x, y - o.y }; }
    REV_HOST_DEVICE vec2<T> operator-(const point2& o)  const { return { x - o.x, y - o.y }; }
};
template<typename T> REV_HOST_DEVICE point2<T> operator*(T s, const point2<T>& v) { return { s * v.x, s * v.y }; }

// 3D types

#define REV_IMPORT_ARRAY3(T, array3) \
    T x, y, z; \
    REV_HOST_DEVICE constexpr array3(T c = 0) : x(c), y(c), z(c) {} \
    REV_HOST_DEVICE constexpr array3(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {} \
    \
    REV_HOST_DEVICE array3 operator*(T s) const { return { x * s, y * s, z * s }; } \
    REV_HOST_DEVICE array3 operator/(T s) const { return { x / s, y / s, z / s }; } \
    REV_HOST_DEVICE array3 operator-() const { return { -x, -y, -z }; } \
    \
    REV_HOST_DEVICE array3 min(const array3& o) const { return { x < o.x ? x : o.x, y < o.y ? y : o.y, z < o.z ? z : o.z }; } \
    REV_HOST_DEVICE array3 max(const array3& o) const { return { x > o.x ? x : o.x, y > o.y ? y : o.y, z > o.z ? z : o.z }; } \
    \
    template <typename U> REV_HOST_DEVICE explicit constexpr array3(const array3<U>& o) \
        : x(static_cast<T>(o.x)), y(static_cast<T>(o.y)), z(static_cast<T>(o.z)) {} \
    \
    std::string to_string() const { return "[" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + "]"; }

/// \private
template <typename T> struct vec3 {
    REV_IMPORT_ARRAY3(T, vec3)

    REV_HOST_DEVICE vec3 operator+(const vec3& o) const { return { x + o.x, y + o.y, z + o.z }; }
    REV_HOST_DEVICE vec3 operator-(const vec3& o) const { return { x - o.x, y - o.y, z - o.z }; }

    REV_HOST_DEVICE T dot(const vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    REV_HOST_DEVICE vec3 cross(const vec3& o) const { return { y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x }; }

    REV_HOST_DEVICE T norm() const { return sqrt(x*x + y*y + z*z); }
    REV_HOST_DEVICE vec3 normalized() const { T len = norm(); return len != 0 ? (*this) / len : (*this); }
};
template <typename T> REV_HOST_DEVICE vec3<T> operator*(T s, const vec3<T>& v) { return { s * v.x, s * v.y, s * v.z }; }

/// \private
template <typename T>
struct point3 {
    REV_IMPORT_ARRAY3(T, point3)

    REV_HOST_DEVICE point3  operator+(const vec3<T>& o) const { return { x + o.x, y + o.y, z + o.z }; }
    REV_HOST_DEVICE point3  operator-(const vec3<T>& o) const { return { x - o.x, y - o.y, z - o.z }; }
    REV_HOST_DEVICE vec3<T> operator-(const point3& o) const { return { x - o.x, y - o.y, z - o.z }; }
};
template <typename T> REV_HOST_DEVICE point3<T> operator*(T s, const point3<T>& v) { return { s * v.x, s * v.y, s * v.z }; }

/// Aliases for common types

using vec2i = vec2<int>;
using vec2f = vec2<float>;
using vec2d = vec2<double>;

using point2i = point2<int>;
using point2f = point2<float>;
using point2d = point2<double>;

using vec3i = vec3<int>;
using vec3f = vec3<float>;
using vec3d = vec3<double>;

using point3i = point3<int>;
using point3f = point3<float>;
using point3d = point3<double>;
    
} // namespace reverie