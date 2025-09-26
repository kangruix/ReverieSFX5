#include "reverie/core/buffer.h"
#include "reverie/core/vector.h"
#include "reverie/core/backend/memory.h"

namespace reverie {

template <typename T>
Buffer<T>::Buffer(size_t size, const Device& dev)
	: m_size(size), m_device(dev) {
	revMalloc((void**) &m_data, m_size * sizeof(T));
}

template <typename T>
Buffer<T>::~Buffer() {
	revFree(m_data);
}

template <typename T>
void Buffer<T>::copy_from(const Buffer<T>& other) {
	if (m_size != other.m_size)
		throw std::runtime_error("Buffer size mismatch");
	
	if (m_device.type == DeviceType::CPU &&
		other.m_device.type == DeviceType::CPU) {
		std::memcpy(m_data, other.m_data, m_size * sizeof(T));
	}
#ifdef REV_ENABLE_CUDA
	else if (m_device.type == DeviceType::CUDA &&
		other.m_device.type == DeviceType::CUDA) {
		revMemcpy(m_data, other.m_data, m_size * sizeof(T),
			DeviceToDevice);
	}
	else if (m_device.type == DeviceType::CPU &&
		other.m_device.type == DeviceType::CUDA) {
		revMemcpy(m_data, other.m_data, m_size * sizeof(T),
			DeviceToHost);
	}
	else if (m_device.type == DeviceType::CUDA &&
		other.m_device.type == DeviceType::CPU) {
		revMemcpy(m_data, other.m_data, m_size * sizeof(T),
			HostToDevice);
	}
#endif
}

template <typename T>
void Buffer<T>::memset(int value) {
	if (m_device.type == DeviceType::CPU) {
		std::memset(m_data, value, m_size * sizeof(T));
	}
#ifdef REV_ENABLE_CUDA
	else if (m_device.type == DeviceType::CUDA) {
		cudaMemset(m_data, value, m_size * sizeof(T));
	}
#endif
}

template class Buffer<int>;
template class Buffer<float>;

template class Buffer<point3f>;
template class Buffer<vec3i>;

} // namespace reverie