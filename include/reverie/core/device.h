#pragma once
#include <string>

namespace reverie {

enum DeviceType : int {
    CPU  = 0,
    CUDA = 1
};
constexpr DeviceType Default = CPU;

struct Device {
    DeviceType type = Default;
    //int index = 0;
};

} // namespace reverie