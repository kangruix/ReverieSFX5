#include "reverie/core/device.h"

namespace reverie {

std::string Device::to_string() const {
	switch (type) {
	case DeviceType::CPU: return "CPU";
#ifdef REV_ENABLE_CUDA
	case DeviceType::CUDA: return "CUDA";
#endif
	default: return "";
	}
}

} // namespace reverie