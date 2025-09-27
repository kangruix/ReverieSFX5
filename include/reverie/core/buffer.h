#pragma once
#include "reverie/core/device.h"
#include <vector>

namespace reverie {

/**
 * \brief Device-aware buffer with memory management
 */
template <typename T>
class Buffer {
public:
    /// Default constructor: create empty buffer
    Buffer() = default;
    
    /// Constructor: create fixed-size buffer (values uninitialized)
    Buffer(size_t size, const Device& dev = Device());

    /// Constructor: initialize from std::vector
    Buffer(const std::vector<T>& vec, const Device& dev = Device());

    /// Copy constructor (deep copy)
    Buffer(const Buffer& other, const Device& dev = Device());

    /// Move constructor
    Buffer(Buffer&& other) noexcept;

    /// Move assignment operator
	Buffer& operator=(Buffer&& other) noexcept;

    /// Destructor
    ~Buffer();

    /// Fills buffer with \c value
    void fill(int value);

    /// \returns CPU-compatible pointer to data
    const T* view() const;

    T* data() { return m_data; }
    const T* data() const { return m_data; }

	size_t size() const { return m_size; }
	const Device& device() const { return m_device; }

    /// Unchecked element access
    T& operator[](size_t i) { return m_data[i]; }
	const T& operator[](size_t i) const { return m_data[i]; }

    std::string to_string(bool device = true) const;

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