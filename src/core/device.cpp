#include "reverie/core/device.h"

namespace reverie {

std::string Device::to_string() const {
	switch (type) {
	case DeviceType::CPU: return "cpu";
	case DeviceType::CUDA: return "cuda";
	default: return "";
	}
}

} // namespace reverie