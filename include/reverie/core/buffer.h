#pragma once
#include "reverie/core/device.h"

namespace reverie {

template <typename T>
class Buffer {
    T* data_;
    size_t size_;
    DeviceType device_;

public:
    Buffer(size_t size = 0, DeviceType type = Default);
    ~Buffer();
};

} // namespace reverie