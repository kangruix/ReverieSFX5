#include "reverie/core/device.h"

namespace reverie {

std::string Device::to_string() const {
	switch (type) {
	case DeviceType::CPU: return "CPU";
	case DeviceType::CUDA: return "CUDA";
	default: return "";
	}
}

} // namespace reverie