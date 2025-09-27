#include "reverie/core/geometry/bbox.h"
#include "reverie/core/backend/kernels.h"
#include <sstream>

namespace reverie {
namespace geometry {

template <> BBox3f compute_bbox<DeviceType::CPU>(
	const Buffer<point3f>& points) {
	assert(points.device().type == DeviceType::CPU);

	BBox3f bbox;
	#pragma omp parallel
	{
		BBox3f l_bbox;  // local BBox
		#pragma omp for nowait
		for (int i = 0; i < points.size(); ++i) {
			l_bbox.extend(points[i]);
		}
		#pragma omp critical
		{
			bbox.extend(l_bbox);
		}
	}
	return bbox;
}

BBox3f compute_bbox(const Buffer<point3f>& points) {
	if (points.device().type == DeviceType::CPU) {
		return compute_bbox<DeviceType::CPU>(points);
	}
#ifdef REV_ENABLE_CUDA
	else if (points.device().type == DeviceType::CUDA) {
		return compute_bbox<DeviceType::CUDA>(points);
	}
#endif
	return BBox3f();
}

template <typename T>
std::string BBox<T>::to_string() const {
	if (!is_valid()) return "BBox[]\n";
	
	std::stringstream ss;
    ss << "BBox<" << typeid(T).name() << ">[\n" <<
		"  min = " << min.to_string() << ",\n" << 
		"  max = " << max.to_string() << "\n" << 
		"]";
	return ss.str();
}

template struct BBox<float>;

} // namespace geometry
} // namespace reverie