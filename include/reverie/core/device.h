#pragma once
#include <string>

namespace reverie {

enum class DeviceType : int {
    CPU = 0,
#ifdef REV_ENABLE_CUDA
    CUDA = 1
#endif
};

#ifdef REV_ENABLE_CUDA
constexpr DeviceType Default = DeviceType::CUDA;
#else
constexpr DeviceType Default = DeviceType::CPU;
#endif

struct Device {
    const DeviceType type = Default;
    //int index = 0;

    std::string to_string() const;
};

} // namespace reverie