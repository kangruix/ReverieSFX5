#include "reverie/core/geometry/bbox.h"
#include <sstream>

namespace reverie {
namespace geometry {

template <> BBox3f compute_bbox<DeviceType::CPU>(
	const point3f* d_points, int num_points) {
	return BBox3f();
}

template <typename T>
std::string BBox<T>::to_string() const {
	if (!is_valid()) return "BBox[]\n";
	
	std::stringstream ss;
    ss << "BBox[\n" <<
		"  min = " << min.to_string() << ",\n" << 
		"  max = " << max.to_string() << "\n" << 
		"]\n";
	return ss.str();
}

template struct BBox<float>;

} // namespace geometry
} // namespace reverie