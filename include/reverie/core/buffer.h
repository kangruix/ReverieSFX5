#pragma once
#include "reverie/core/device.h"
#include "reverie/core/vector.h"

namespace reverie {

template <typename T>
class Buffer {
public:
    Buffer() = default;
    Buffer(size_t size, const Device& dev = Default);

    ~Buffer();
	void copy_from(const Buffer<T>& other);
    void memset(int value);

    const T* get() const { return m_data; }
	size_t size() const { return m_size; }

private:
    T* m_data = nullptr;
    size_t m_size = 0;
    Device m_device;
};

} // namespace reverie