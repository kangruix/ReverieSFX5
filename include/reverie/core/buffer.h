#pragma once
#include "reverie/core/device.h"
#include <vector>

namespace reverie {

template <typename T>
class Buffer {
public:
    /// Default constructor: create empty buffer
    Buffer() = default;
    
    /// Constructor: create buffer on device (values uninitialized)
    Buffer(size_t size, const Device& dev = { Default });

    /// Constructor: initialize from std::vector
    Buffer(const std::vector<T>& vec, const Device& dev = { Default });

    /// Copy constructor (deep copy)
    Buffer(const Buffer<T>& other, const Device& dev = { Default });

    /// Move constructor
    Buffer(Buffer<T>&& other) noexcept;

    /// Move assignment operator
	Buffer& operator=(Buffer<T>&& other) noexcept;

    /// Destructor
    ~Buffer();

    /// Fills buffer with \c value
    void fill(int value);

    /// \returns CPU-compatible pointer to data
    const T* view() const;

    const T* data() const { return m_data; }
	size_t size() const { return m_size; }
	const Device& device() const { return m_device; }

    /// Unchecked element access
	const T& operator[](size_t i) const { return m_data[i]; }

private:
    T* m_data = nullptr;
    size_t m_size = 0;
    Device m_device;

    T* m_host_data = nullptr;

    void allocate();
    void deallocate();

    void copy_from(const T* src_ptr, DeviceType src_dev);
};

} // namespace reverie